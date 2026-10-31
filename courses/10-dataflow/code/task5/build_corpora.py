
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DataFlow Task5 — 构建高质量训练数据：设计语料构造器
在 vm-60 上运行：python build_corpora.py --out <dir>

产出两份带缺陷清单的输入语料（每行 {"id": ..., "raw_content"/"instruction"...}）：
  pt_input.jsonl   —— 预训练数据过滤实验：干净文档 + 每类缺陷文档 + 边界对照文档
  sft_input.jsonl  —— SFT 数据过滤实验：干净指令对 + 缺陷指令对 + 边界对照
设计原则：每条文档标注 defect（预期触发的过滤器）与 expect（keep/drop），
用于事后核对「漏斗的每一刀砍在了谁身上」。文档内容全部本地撰写/程序生成，不下载外部数据。
"""
import argparse
import json
import os
import random

# ---------------------------------------------------------------- PT 语料
# 干净文档：真实英文知识性段落（≥20 词、≥3 句、平均词长 3-10、标点充足、无特殊字符）
CLEAN_PT = [
    "Thermodynamics is a branch of physics that deals with heat, work, and temperature, and their relation to energy. The behavior of these quantities is governed by the four laws of thermodynamics. These laws convey a quantitative description using measurable macroscopic physical quantities. They forbid phenomena such as perpetual motion.",
    "Cellular respiration is a set of metabolic reactions that convert biochemical energy from nutrients into adenosine triphosphate. The process takes place inside the cells of living organisms. Reactions involved in respiration are catabolic, breaking large molecules into smaller ones and releasing energy. Respiration is one of the key ways a cell releases chemical energy.",
    "Photosynthesis is a system of biological processes by which photosynthetic organisms convert light energy into chemical energy. This energy is later released to fuel the activities of the organism. The process uses carbon dioxide and water as raw materials and releases oxygen as a byproduct. Most plants, algae, and cyanobacteria perform photosynthesis.",
    "Plate tectonics describes the large-scale motion of seven major plates and many minor plates of the lithosphere. The theory built on the concept of continental drift, which was developed during the first decades of the twentieth century. Earthquakes, volcanic activity, and mountain building occur along plate boundaries. The lateral movement of plates is typically measured in centimeters per year.",
    "A neural network is a group of connected units called neurons that send signals to one another. Each neuron receives inputs, applies a weighted sum followed by a nonlinear activation, and passes the result onward. Training adjusts the weights so that the network output approximates a desired function. Modern networks may contain billions of parameters.",
    "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape from them. The boundary of no return is called the event horizon. Although black holes have a great effect on the fate of objects crossing the horizon, they themselves have no locally detectable features. Astronomers infer their presence from the radiation emitted by matter falling inward.",
    "Deoxyribonucleic acid is a polymer composed of two polynucleotide chains that coil around each other to form a double helix. The polymer carries genetic instructions for the development and functioning of known organisms. Each strand is a sequence of nucleotides built from four bases. The two strands store the same biological information.",
    "Osmosis is the spontaneous net movement of solvent molecules through a selectively permeable membrane. The movement proceeds from a region of higher solvent potential to a region of lower solvent potential. It equalizes solute concentrations on the two sides of the membrane. Osmosis is of vital importance in biological systems.",
    "Electromagnetism is an interaction that occurs between particles with electric charge via electromagnetic fields. The electromagnetic force is one of the four fundamental forces of nature. It combines the effects of electricity and magnetism into a single theory. Oscillating electric and magnetic fields propagate through space as electromagnetic radiation.",
    "Evolution is change in the heritable characteristics of biological populations over successive generations. These characteristics are the expressions of genes passed from parent to offspring. Genetic variation arises from mutation and recombination, and evolutionary processes such as selection act upon this variation. The biodiversity of life is the product of billions of years of evolution.",
    "The water cycle, also known as the hydrologic cycle, describes the continuous movement of water within the Earth and atmosphere. It involves the major processes of evaporation, transpiration, condensation, precipitation, and runoff. The mass of water on Earth remains fairly constant over time. However, the partitioning of water into ice, fresh water, and salt water shifts with climate.",
    "An operating system is system software that manages computer hardware and software resources. It provides common services for computer programs, including scheduling, memory management, and device control. Time-sharing operating systems schedule tasks for efficient use of the system. They may also include accounting software for cost allocation of processor time.",
]

# 边界对照组（贴着阈值构造，验证阈值语义）
BOUNDARY_PT = [
    # 恰好 20 个词且 3 句（WordNumberFilter min_words=20 与 SentenceNumberFilter min_sentences=3 的保留边界）
    ("boundary_words_20",
     "the quick brown fox jumps over. a lazy dog sleeps nearby. this document contains exactly twenty words in three sentences here now.",
     "keep"),
    # 19 个词且 3 句（应被 WordNumberFilter 砍掉）
    ("boundary_words_19",
     "the quick brown fox jumps over. a lazy dog sleeps nearby. this document contains nineteen words in three sentences.",
     "drop"),
    # 恰好 3 句（SentenceNumberFilter min_sentences=3 的保留边界）
    ("boundary_sentences_3",
     "The first sentence of this document states a fact. A second sentence adds supporting detail here. The final sentence concludes the thought completely.",
     "keep"),
    # 2 句但 23 词（应被 SentenceNumberFilter 砍掉，而不是 WordNumberFilter）
    ("boundary_sentences_2",
     "Only two sentences appear in this short document for testing purposes here. This is the second and final sentence of the document.",
     "drop"),
]

# 缺陷文档：每条形靶向一个过滤器
DEFECT_PT = [
    # 精确重复（MinHash 去重保留首条、砍掉第二条）
    ("dup_exact_a", CLEAN_PT[0], "drop_later"),
    ("dup_exact_b", CLEAN_PT[0], "drop"),
    # 近似重复：在 64 词干净文档上只替换 1 个词（5-gram Jaccard≈0.92 > 0.9）
    ("near_dup", CLEAN_PT[5].replace("gravity", "gravitation"), "drop"),
    # 过短：<20 词
    ("too_short", "This document is far too short to survive the word number filter.", "drop"),
    # 以冒号结尾（ColonEndFilter）
    ("colon_end", "Photosynthesis converts light energy into chemical energy inside the chloroplasts of plant cells. The light dependent reactions capture photons and split water molecules. The calvin cycle then fixes carbon dioxide into sugars. The overall inputs and outputs can be summarized as follows:", "drop"),
    # 句子数不足：1 句
    ("one_sentence", "A single long sentence without any terminal punctuation until the very end of this rather lengthy document that keeps going and going", "drop"),
    # 省略号行比例过高（>=30% 行以 ... 结尾）
    ("ellipsis_lines", "The report was cut off mid sentence...\nThe second line also ends with an ellipsis...\nThe third line does the same thing again...\nOnly this final line ends normally.", "drop"),
    # 空内容（ContentNullFilter）
    ("null_content", "   ", "drop"),
    # 平均词长过长（MeanWordLengthFilter max 10）：3 句化学命名长词
    ("mean_len_long", "Spectroelectrochemistry characterizes electrochromism chemiluminescence chemiluminescence chemiluminescence chemiluminescence chemiluminescence chemiluminescence chemiluminescence. Chemiluminescence chemiluminescence chemiluminescence chemiluminescence chemiluminescence chemiluminescence chemiluminescence chemiluminescence chemiluminescence chemiluminescence. Chemiluminescence chemiluminescence chemiluminescence chemiluminescence chemiluminescence chemiluminescence chemiluminescence chemiluminescence chemiluminescence.", "drop"),
    # 平均词长过短（<3）：3 句短词文档
    ("mean_len_short", "a b c d e f g h. i j k l m n o p. q r s t u v w x.", "drop"),
    # 符号比过高（SymbolWordRatioFilter >= 0.4）：大量 #（保持词长/句数/词数合规，只让符号比超标）
    ("symbol_heavy", "data data data data data data data. data data data data data data data. data data data data data. ####################.", "drop"),
    # HTML 实体（HtmlEntityFilter）
    ("html_entity", "The website rendered the quotation as &quot;smart quotes&quot; and the ampersand as &amp; in the exported page. Editors noticed that &nbsp; characters appeared between words. The pipeline stored the raw entities instead of the decoded characters. This broke downstream text matching for search.", "drop"),
    # 超长无标点段落（NoPuncFilter：句内 >112 词）：前两句正常，第三句 120 词无标点
    ("no_punc", "This is the first normal sentence of the document. Here is a second sentence that ends properly. " + " ".join(["word"] * 120) + ".", "drop"),
    # 特殊字符（SpecialCharacterFilter：含 &#247; / {/U} / u200e 等字面模式）
    ("special_char", "The export contained mojibake where division signs appeared as &#247; and placeholder blocks showed {/U} in the text. Some records included the literal marker u200e between words. These artifacts indicate a broken decoding pipeline upstream. Cleaning them requires re-exporting from the source system.", "drop"),
    # 水印词（WatermarkFilter）
    ("watermark", "This internal document is marked Confidential and must not leave the building. The footer carries a Copyright notice belonging to the publisher. Staff received a reminder about the Watermark printed on every page. Distribution outside the company is strictly prohibited by policy.", "drop"),
    # 花括号比例过高（CurlyBracketFilter >= 0.025）
    ("curly_bracket", "The configuration payload looks like {\"alpha\": 1, \"beta\": {\"gamma\": 2, \"delta\": {\"epsilon\": 3, \"zeta\": {\"eta\": 4, \"theta\": {\"iota\": 5, \"kappa\": {\"lambda\": 6, \"mu\": 7}}}}}} and nested objects keep appearing everywhere in the log. {\"nu\": 8, \"xi\": {\"omicron\": 9, \"pi\": {\"rho\": 10}}}. {\"sigma\": 11, \"tau\": {\"upsilon\": 12}}.", "drop"),
    # 大写词比例过高（CapitalWordsFilter > 0.2）：3 句缩略语
    ("capital_words", "NASA FBI CIA NSA DHS ICE ATF DEA. TSA DOD DOJ HHS HUD DOE. DOI USDA EPA FDA FCC SEC.", "drop"),
    # Lorem ipsum 占位文本（LoremIpsumFilter）
    ("lorem_ipsum", "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore. The template page still contains the placeholder paragraph instead of the real article body. Et dolore magna aliqua ut enim ad minim veniam. Designers forgot to replace the dummy text before publishing the page.", "drop"),
    # 唯一词比例过低（UniqueWordsFilter <= 0.1）：整段重复同句
    ("unique_words_low", ("The system processes the data records. " * 30).strip(), "drop"),
    # 项目符号行比例过高（LineStartWithBulletpointFilter > 0.9）
    ("bullet_lines", "\u2022 First item of the unordered list appears here.\n\u2022 Second item of the unordered list appears here.\n\u2022 Third item of the unordered list appears here.\n\u2022 Fourth item of the unordered list appears here.\n\u2022 Fifth item of the unordered list appears here.", "drop"),
    # javascript 行过多（LineWithJavascriptFilter：非 js 行数 < 3 且总行数 > 3）
    ("javascript_lines", "Enable javascript in your browser settings to continue.\nThis site requires javascript to render the interactive chart.\nPlease enable javascript and reload this page now.\nJavascript is disabled in your current browser profile.\nWithout javascript the application cannot load.", "drop"),
    # 屏蔽词（BlocklistFilter：>1 个屏蔽词）
    ("blocklist", "The argument escalated quickly and he called the man a bastard twice over. What a bastard thing to do, she replied. The meeting ended in shouting. Nobody could believe the bastard behaviour on display that afternoon.", "drop"),
    # 非英文（LanguageFilter：fasttext 判为非 eng_Latn；西班牙语，PairQual-en 得分 2.02 可过质量关，
    # 保证是被语言过滤器而不是质量过滤器砍掉——实测法语/德语文本过不了 PairQual-en 的 min_score=2）
    ("non_english", "El sistema nervioso central esta formado por el encefalo y la medula espinal. Procesa la informacion sensorial, controla las funciones motoras y hace posibles las funciones cognitivas superiores. Las neuronas se comunican entre si mediante sinapsis quimicas y electricas. Los estudios modernos de neurociencia han ampliado profundamente la comprension de estos mecanismos en las ultimas decadas.", "drop"),
    # 低质量 SEO 垃圾文本（PairQual 低分候选；小写化以免先被 CapitalWordsFilter 砍掉）
    ("seo_spam", "click here now for the best deals online today only. you will not want to miss this amazing offer available for a limited time. act fast before it is gone forever and never available again. this incredible deal will change your life starting today. buy now and save big money instantly online.", "drop"),
    # 随机乱码文本（PairQual 低分候选；补齐句号以免先被 SentenceNumberFilter 砍掉）
    ("gibberish", "xkcd qwerty uiop asdf ghjkl. zxcv bnm mnbvcx poiuy trewq. lkjh gfdsa poiuyt rewsx cvbnm mlkjh gfdsaqwerty uiopasdf ghjklzxcv bnm mnbvcx poiuytrewq lkjhgfdsa poiuytrewq asdfghjkl.", "drop"),
]

# 会被 refiner 改写而非删除的文档（验证 refine 类算子只改内容不过滤）
REFINE_PT = [
    ("emoji_doc", "The public celebration of the launch featured fireworks displays and parades across the city center. Historians record that fireworks have marked public celebrations in Europe since the seventeenth century. The product was released to the general public the following morning. Support infrastructure remained stable throughout the first week of operation. The evening concluded with a spectacular fireworks show \U0001F386 that drew thousands of spectators to the riverfront.", "keep"),
    ("html_url_doc", "The complete technical report of the annual survey is published online together with supplementary materials. An identical mirror of the document is maintained at a secondary site for redundancy. Charts and tables accompany both releases of the report. Readers are asked to cite the original source when quoting figures. See https://example.com/reports/2026/annual-summary and http://mirror.example.org/report.html for access.", "keep"),
    ("extra_spaces_doc", "The    laboratory   measured   the   sample   three   times   for    accuracy   with   calibrated instruments.   Results   were   consistent   across   all   trials   and   instruments.   The    final   report    was   published   online   after   peer   review.    Additional   measurements   were   collected   during   the   following   semester.", "keep"),
]

# ---------------------------------------------------------------- SFT 语料
CLEAN_SFT = [
    ("sft_clean_01",
     "Provide a detailed comparison between the 'list' and 'tuple' data structures in Python, focusing on mutability, performance, and common use cases.",
     "Certainly. The primary distinction between lists and tuples in Python lies in their mutability. Lists are mutable, meaning their elements can be added, removed, or modified after creation. Tuples are immutable; once created, their contents cannot change. This difference has performance implications: tuples are slightly faster to create and iterate, and they can be used as dictionary keys or stored in sets because they are hashable. Lists, by contrast, support a richer set of methods such as append, extend, and sort. In practice, lists are preferred when you need a dynamic collection whose size may change, while tuples are used for fixed, heterogeneous records such as coordinates or database rows. Choosing the right one communicates intent and can prevent accidental modification of data that should remain constant."),
    ("sft_clean_02",
     "Explain the process of natural selection using the example of the peppered moth during the Industrial Revolution.",
     "Before the Industrial Revolution, light-colored peppered moths were well camouflaged against lichen-covered trees in England. Pollution killed the lichens and covered the bark with soot, making the dark background favorable for dark-colored moths, which were then better hidden from predatory birds. Because birds ate the more visible moths, the dark variant survived and reproduced at higher rates, and its frequency in the population rose sharply. When pollution later declined, the light form again became advantageous. This classic case illustrates how environmental change alters the selective pressures acting on heritable variation, and it remains one of the most documented examples of natural selection in action."),
    ("sft_clean_03",
     "Write a short paragraph explaining what an API is to someone with no programming background.",
     "An API, or application programming interface, is a way for two programs to talk to each other, much like a waiter carries your order to the kitchen and brings back the food. Instead of knowing how the kitchen works, you just use the menu. In the same way, one program can request information or an action from another program without understanding its internal details. For example, a weather app on your phone asks a remote server for the forecast through an API, and the server sends back the data in a format the app knows how to display."),
    ("sft_clean_04",
     "List five practical habits that improve sleep quality, and briefly explain the mechanism behind each.",
     "First, keep a consistent sleep and wake time, because a regular schedule stabilizes the circadian rhythm that controls alertness. Second, reduce bright light and screen exposure in the hour before bed, since light suppresses melatonin production. Third, avoid caffeine after mid-afternoon, as its half-life means a late coffee can still block adenosine receptors at bedtime. Fourth, keep the bedroom cool and dark, which supports the drop in core body temperature that accompanies sleep onset. Fifth, establish a short wind-down routine such as reading or stretching, which signals safety to the nervous system and shortens the time it takes to fall asleep."),
    ("sft_clean_05",
     "What is the difference between supervised and unsupervised learning? Give one concrete example of each.",
     "Supervised learning trains a model on examples that include the correct answer, so the model learns a mapping from inputs to known outputs; for instance, labeling thousands of emails as spam or not spam and training a classifier to predict that label. Unsupervised learning works with data that has no labels and instead discovers structure, such as grouping customers into segments based on purchasing behavior without being told what the segments mean. The key difference is whether the training signal comes from explicit target values or from patterns in the data itself. Many practical systems combine both, for example by pretraining without labels and then fine-tuning with labeled examples."),
    ("sft_clean_06",
     "Explain the concept of compound interest with a simple numerical example.",
     "Compound interest means that you earn interest not only on your original deposit but also on the interest that has already accumulated. Suppose you invest one thousand dollars at an annual rate of ten percent, compounded once per year. After the first year you have one thousand one hundred dollars. In the second year the ten percent applies to the full eleven hundred dollars, giving one thousand two hundred ten dollars. After ten years the balance grows to about two thousand five hundred ninety four dollars, which is substantially more than the two thousand dollars you would have with simple interest, and the gap widens every additional year."),
    ("sft_clean_07",
     "How does a vaccine protect the body from disease?",
     "A vaccine trains the immune system in advance by presenting a harmless piece or mimic of a pathogen, such as an inactivated virus, a protein fragment, or genetic instructions for making one. This exposure prompts the immune system to produce antibodies and memory cells specific to that pathogen. If the real pathogen is encountered later, the memory cells recognize it quickly and mount a faster, stronger response that often stops the infection before it causes serious illness. Because protection comes from this trained memory rather than from the vaccine itself remaining in the body, the effect can last for months, years, or even a lifetime depending on the disease."),
    ("sft_clean_08",
     "Describe three common causes of software bugs and how to prevent them.",
     "One common cause is incorrect assumptions about input, which is best prevented by validating and sanitizing data at system boundaries. A second cause is shared mutable state, especially in concurrent code, which can be reduced by favoring immutable data structures and clear ownership of state. A third cause is unclear interfaces between components, which can be addressed with explicit contracts, tests, and documentation. In practice, most bugs also reflect a gap in understanding, so writing focused tests and reviewing code with another person are among the most reliable prevention strategies available."),
]

DEFECT_SFT = [
    # output 过短（WordNumberFilter min_words=20）
    ("sft_short_output",
     "Explain what a database index is.",
     "It makes queries faster.",
     "drop"),
    # output 过长（>1000 词）
    ("sft_long_output",
     "Describe the water cycle in detail.",
     "The water cycle moves water across the planet continuously. " * 260,
     "drop"),
    # output 为空
    ("sft_empty_output",
     "Explain the theory of relativity.",
     "",
     "drop"),
    # output 高度重复（Superfiltering IFD 极低 + Deita 低分候选）
    ("sft_repetitive_output",
     "Explain why the sky is blue.",
     "The sky is blue because of scattering. " * 40,
     "drop"),
    # instruction 与 output 不对应（Superfiltering IFD 高 → 指令无法预测回答）
    ("sft_mismatch",
     "What is the boiling point of water at sea level?",
     "The Roman Empire reached its greatest territorial extent under the emperor Trajan in the early second century, when its provinces stretched from Britain to Mesopotamia.",
     "drop"),
    # 低质量 but 结构完整的回答（Deita 低分候选）
    ("sft_low_quality",
     "How do airplanes fly?",
     "Airplanes fly because of wings. The wings make them go up. Pilots control the plane. It is complicated. Engines help. The sky is big. Fuel is needed. That is basically it.",
     "drop"),
    # 边界：output 恰好 20 词
    ("sft_boundary_20",
     "Summarize the causes of the French Revolution.",
     "The revolution grew from fiscal crisis, social inequality, Enlightenment ideas, and weak leadership, which together produced widespread unrest across the country.",
     "keep"),
    # 边界：output 19 词
    ("sft_boundary_19",
     "Summarize the causes of the French Revolution.",
     "The revolution grew from fiscal crisis, social inequality, Enlightenment ideas, and weak leadership, producing widespread unrest across the country.",
     "drop"),
]


def write_jsonl(path, rows, key):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[write] {path}: {len(rows)} rows (key={key})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    ap.add_argument("--seed", type=int, default=20261101)
    args = ap.parse_args()
    random.seed(args.seed)

    # ---- PT 语料：干净 + 边界 + 缺陷 + refiner 演示，id 即清单
    pt_rows = []
    manifest_pt = []
    for i, text in enumerate(CLEAN_PT):
        rid = f"pt_clean_{i:02d}"
        pt_rows.append({"id": rid, "raw_content": text})
        manifest_pt.append({"id": rid, "defect": "none", "expect": "keep"})
    for rid, text, expect in BOUNDARY_PT:
        pt_rows.append({"id": rid, "raw_content": text})
        manifest_pt.append({"id": rid, "defect": "boundary", "expect": expect})
    for rid, text, expect in DEFECT_PT:
        pt_rows.append({"id": rid, "raw_content": text})
        manifest_pt.append({"id": rid, "defect": rid, "expect": "drop" if expect.startswith("drop") else "keep"})
    for rid, text, expect in REFINE_PT:
        pt_rows.append({"id": rid, "raw_content": text})
        manifest_pt.append({"id": rid, "defect": "refine_demo", "expect": "keep"})

    # dup_exact_a 保留（首条）、dup_exact_b 删除：修正 manifest 语义
    for m in manifest_pt:
        if m["id"] == "dup_exact_a":
            m["expect"] = "keep"
            m["defect"] = "duplicate_first_copy"

    write_jsonl(os.path.join(args.out, "pt_input.jsonl"), pt_rows, "raw_content")

    # ---- SFT 语料
    sft_rows = []
    manifest_sft = []
    for rid, ins, out in CLEAN_SFT:
        sft_rows.append({"id": rid, "instruction": ins, "input": "", "output": out})
        manifest_sft.append({"id": rid, "defect": "none", "expect": "keep"})
    for item in DEFECT_SFT:
        rid, ins, out, expect = item
        sft_rows.append({"id": rid, "instruction": ins, "input": "", "output": out})
        manifest_sft.append({"id": rid, "defect": rid, "expect": expect})

    write_jsonl(os.path.join(args.out, "sft_input.jsonl"), sft_rows, "instruction/output")

    with open(os.path.join(args.out, "manifest_pt.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_pt, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "manifest_sft.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_sft, f, ensure_ascii=False, indent=2)
    print(f"[write] manifest_pt.json: {len(manifest_pt)} rows")
    print(f"[write] manifest_sft.json: {len(manifest_sft)} rows")


if __name__ == "__main__":
    main()
