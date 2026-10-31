
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DataFlow Task5 — 预训练数据过滤（PT text filter pipeline，官方 GPU 版改造，vm-60 实测）

官方原型：dataflow/statics/pipelines/gpu_pipelines/text_pt_filter.py（PTTextFilter_GPUPipeline）
改造点（相对官方）：
  1) first_entry_file_name 改为 T5_INPUT（本任务设计语料 pt_input.jsonl，含 id 列，便于 id 级核对）；
  2) cache_path/prefix 改为 T5_CACHE/T5_PREFIX；
  3) 模型缓存统一指向 T5_HF_CACHE（/mnt/nfs/task5_hf，离线预置 PairQual/fasttext 权重）；
  4) 算子、参数与执行顺序与官方 GPU 版完全一致：
     refiners(emoji/url/spaces) -> blocklist -> word_number -> colon_end -> sentence_number ->
     line_end_ellipsis -> content_null -> mean_word_length -> symbol_word_ratio -> html_entity ->
     no_punc -> special_character -> watermark -> curly_bracket -> capital_words -> lorem_ipsum ->
     unique_words -> char_number -> line_start_bulletpoint -> line_with_javascript ->
     PairQual(min_score=2) -> LanguageFilter(eng_Latn) -> MinHashDedup(threshold=0.9, 5-gram)
  5) 每步打印 in/out 行数与耗时，用于重建过滤漏斗（verify_task5.py 再做 id 级核对）。
"""
import os
import time

from dataflow.operators.general_text import (
    MinHashDeduplicateFilter,
    LanguageFilter,
    WordNumberFilter,
    BlocklistFilter,
    ColonEndFilter,
    SentenceNumberFilter,
    LineEndWithEllipsisFilter,
    ContentNullFilter,
    MeanWordLengthFilter,
    SymbolWordRatioFilter,
    HtmlEntityFilter,
    NoPuncFilter,
    SpecialCharacterFilter,
    WatermarkFilter,
    CurlyBracketFilter,
    CapitalWordsFilter,
    LoremIpsumFilter,
    UniqueWordsFilter,
    CharNumberFilter,
    LineStartWithBulletpointFilter,
    LineWithJavascriptFilter,
    RemoveExtraSpacesRefiner,
    RemoveEmojiRefiner,
    HtmlUrlRemoverRefiner,
)
from dataflow.operators.text_pt import PairQualFilter
from dataflow.utils.storage import FileStorage


def count_lines(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


class PTTextFilter_Task5:
    def __init__(self):
        self.input_path = os.environ.get("T5_INPUT") or "./pt_input.jsonl"
        self.cache_path = os.environ.get("T5_CACHE") or "./cache_pt"
        self.prefix = os.environ.get("T5_PREFIX") or "pt_filter_task5"
        os.makedirs(self.cache_path, exist_ok=True)
        self.storage = FileStorage(
            first_entry_file_name=self.input_path,
            cache_path=self.cache_path,
            file_name_prefix=self.prefix,
            cache_type="jsonl",
        )
        self.model_cache_dir = os.environ.get("T5_HF_CACHE") or "/mnt/nfs/task5_hf"

        # —— 与官方 GPU 版逐一对应 ——
        self.remove_extra_spaces_refiner = RemoveExtraSpacesRefiner()
        self.remove_emoji_refiner = RemoveEmojiRefiner()
        self.html_remove_refiner = HtmlUrlRemoverRefiner()
        self.minhash_deduplicator = MinHashDeduplicateFilter(num_perm=128, threshold=0.9, use_n_gram=True, ngram=5)
        self.blocklist_filter = BlocklistFilter()
        self.word_number_filter = WordNumberFilter(min_words=20, max_words=100000)
        self.colon_end_filter = ColonEndFilter()
        self.sentence_number_filter = SentenceNumberFilter(min_sentences=3, max_sentences=7500)
        self.line_end_with_ellipsis_filter = LineEndWithEllipsisFilter(threshold=0.3)
        self.content_null_filter = ContentNullFilter()
        self.mean_word_length_filter = MeanWordLengthFilter(min_length=3, max_length=10)
        self.symbol_word_ratio_filter = SymbolWordRatioFilter(threshold=0.4)
        self.html_entity_filter = HtmlEntityFilter()
        self.no_punc_filter = NoPuncFilter(threshold=112)
        self.special_character_filter = SpecialCharacterFilter()
        self.watermark_filter = WatermarkFilter(watermarks=['Copyright', 'Watermark', 'Confidential'])
        self.curly_bracket_filter = CurlyBracketFilter(threshold=0.025)
        self.capital_words_filter = CapitalWordsFilter(threshold=0.2, use_tokenizer=False)
        self.lorem_ipsum_filter = LoremIpsumFilter(threshold=3e-8)
        self.unique_words_filter = UniqueWordsFilter(threshold=0.1)
        self.char_number_filter = CharNumberFilter(threshold=100)
        self.line_start_with_bulletpoint_filter = LineStartWithBulletpointFilter(threshold=0.9)
        self.line_with_javascript_filter = LineWithJavascriptFilter(threshold=3)
        # 官方 GPU 版未传 model_cache_dir（用默认相对路径 ./dataflow_cache，会现拉权重）；
        # 这里显式指向离线预置缓存（改造点）
        self.quality_filter = PairQualFilter(min_score=2, max_score=10000, lang='en', model_cache_dir=self.model_cache_dir)
        self.language_filter = LanguageFilter(allowed_languages='__label__eng_Latn', model_cache_dir=self.model_cache_dir)

    def run_op(self, name, fn, **kwargs):
        step_storage = self.storage.step()
        n = step_storage.operator_step
        in_path = self.input_path if n == 0 else os.path.join(self.cache_path, f"{self.prefix}_step{n}.jsonl")
        out_path = os.path.join(self.cache_path, f"{self.prefix}_step{n + 1}.jsonl")
        before = count_lines(in_path)
        t0 = time.time()
        fn(storage=step_storage, **kwargs)
        after = count_lines(out_path)
        print(f"[STEP] {name:32s} {str(before):>4s} -> {str(after):>4s}  ({time.time()-t0:6.1f}s)", flush=True)

    def forward(self):
        self.run_op("RemoveEmojiRefiner", self.remove_emoji_refiner.run, input_key="raw_content")
        self.run_op("HtmlUrlRemoverRefiner", self.html_remove_refiner.run, input_key="raw_content")
        self.run_op("RemoveExtraSpacesRefiner", self.remove_extra_spaces_refiner.run, input_key="raw_content")
        self.run_op("BlocklistFilter", self.blocklist_filter.run, input_key='raw_content')
        self.run_op("WordNumberFilter", self.word_number_filter.run, input_key='raw_content')
        self.run_op("ColonEndFilter", self.colon_end_filter.run, input_key='raw_content')
        self.run_op("SentenceNumberFilter", self.sentence_number_filter.run, input_key='raw_content')
        self.run_op("LineEndWithEllipsisFilter", self.line_end_with_ellipsis_filter.run, input_key='raw_content')
        self.run_op("ContentNullFilter", self.content_null_filter.run, input_key='raw_content')
        self.run_op("MeanWordLengthFilter", self.mean_word_length_filter.run, input_key='raw_content')
        self.run_op("SymbolWordRatioFilter", self.symbol_word_ratio_filter.run, input_key='raw_content')
        self.run_op("HtmlEntityFilter", self.html_entity_filter.run, input_key='raw_content')
        self.run_op("NoPuncFilter", self.no_punc_filter.run, input_key='raw_content')
        self.run_op("SpecialCharacterFilter", self.special_character_filter.run, input_key='raw_content')
        self.run_op("WatermarkFilter", self.watermark_filter.run, input_key='raw_content')
        self.run_op("CurlyBracketFilter", self.curly_bracket_filter.run, input_key='raw_content')
        self.run_op("CapitalWordsFilter", self.capital_words_filter.run, input_key='raw_content')
        self.run_op("LoremIpsumFilter", self.lorem_ipsum_filter.run, input_key='raw_content')
        self.run_op("UniqueWordsFilter", self.unique_words_filter.run, input_key='raw_content')
        self.run_op("CharNumberFilter", self.char_number_filter.run, input_key='raw_content')
        self.run_op("LineStartWithBulletpointFilter", self.line_start_with_bulletpoint_filter.run, input_key='raw_content')
        self.run_op("LineWithJavascriptFilter", self.line_with_javascript_filter.run, input_key='raw_content')
        self.run_op("PairQualFilter", self.quality_filter.run, input_key='raw_content')
        self.run_op("LanguageFilter", self.language_filter.run, input_key="raw_content")
        self.run_op("MinHashDeduplicateFilter", self.minhash_deduplicator.run, input_key='raw_content')


if __name__ == "__main__":
    print("[CONFIG] input    =", os.environ.get("T5_INPUT"), flush=True)
    print("[CONFIG] cache    =", os.environ.get("T5_CACHE"), flush=True)
    print("[CONFIG] hf_cache =", os.environ.get("T5_HF_CACHE"), flush=True)
    t0 = time.time()
    model = PTTextFilter_Task5()
    print(f"[INIT] operators ready ({time.time()-t0:.1f}s)", flush=True)
    model.forward()
    print(f"[DONE] total {time.time()-t0:.1f}s", flush=True)
