#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataFlow Task4 —— Text2SQL 数据优化（refine）流水线（最小改造版）

基于 open-dataflow 1.0.11 官方 api_pipelines/text2sql_pipeline_refine.py，改动点：
  1. db_root_path 指向本机已解压的 SQLite 数据库目录（3 个库）；
  2. 输入改为自建的 pipeline_refine.jsonl 子集（--input）；
  3. LLM 改为 AMD Radeon OpenAI 兼容端点 DeepSeek-V4-Flash（APILLMServing_request）；
  4. embedding 改为本地 bge-small-zh-v1.5（LocalEmbeddingServing, cuda）；
  5. 生成/采样数量按时间预算缩小（num_variations / question_candidates_num /
     CoT sampling_num / SQLExecutionClassifier num_generations），理由见 REPORT；
  6. 每步结束把 cache jsonl 的行数/字段落进 stats JSON，便于证据追溯。

不使用自动下载数据库逻辑（远端 / 盘空间紧张），db_root 必须显式给。
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd

from dataflow import get_logger
from dataflow.operators.text2sql import (
    SQLVariationGenerator,
    Text2SQLQuestionGenerator,
    Text2SQLPromptGenerator,
    Text2SQLCoTGenerator,
    Text2SQLCoTVotingGenerator,
    SQLExecutabilityFilter,
    Text2SQLCorrespondenceFilter,
    SQLComponentClassifier,
    SQLExecutionClassifier,
)
from dataflow.prompts.text2sql import (
    Text2SQLCorrespondenceFilterPrompt,
    Text2SQLCotGeneratorPrompt,
    Text2SQLQuestionGeneratorPrompt,
    SQLVariationGeneratorPrompt,
    Text2SQLPromptGeneratorPrompt,
)
from dataflow.serving import APILLMServing_request, LocalEmbeddingServing
from dataflow.utils.storage import FileStorage
from dataflow.utils.text2sql.database_manager import DatabaseManager

DEFAULT_DB_ROOT = "/home/simin/dataflow-task4/db/downloaded_databases/databases"
DEFAULT_API_BASE = "https://developer.amd.com.cn/radeon/api/v1"
DEFAULT_MODEL = "DeepSeek-V4-Flash"

KEY_FIELDS = [
    "db_id", "SQL", "question", "evidence", "prompt", "cot_reasoning",
    "sql_variation_type", "sql_component_difficulty", "sql_execution_difficulty",
]


def find_embedding_model() -> str:
    """定位本地 HF 缓存里的 bge-small-zh-v1.5 snapshot 目录。"""
    env_path = os.environ.get("EMBEDDING_MODEL_PATH", "").strip()
    if env_path and os.path.isdir(env_path):
        return env_path
    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/*/"
    )
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            "找不到本地 bge-small-zh-v1.5 缓存，请设置 EMBEDDING_MODEL_PATH"
        )
    return candidates[-1].rstrip("/")


class Text2SQLRefineTask4:
    def __init__(self, args):
        self.args = args
        self.logger = get_logger()
        self.stats = []

        db_root = os.path.abspath(args.db_root)
        if not os.path.isdir(db_root):
            raise FileNotFoundError(f"数据库目录不存在: {db_root}")

        input_path = os.path.abspath(args.input)
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"输入 jsonl 不存在: {input_path}")
        self.input_path = input_path
        self.cache_dir = os.path.abspath(args.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        api_base = os.environ.get("AMD_API_BASE", DEFAULT_API_BASE).rstrip("/")
        api_url = api_base + "/chat/completions"
        model_name = os.environ.get("DF_MODEL", DEFAULT_MODEL)
        self.logger.info(f"LLM api_url={api_url} model={model_name}")
        self.llm_serving = APILLMServing_request(
            api_url=api_url,
            model_name=model_name,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            connect_timeout=15.0,
            read_timeout=240.0,
            temperature=args.temperature,
        )

        embedding_model = args.embedding_model or find_embedding_model()
        self.logger.info(f"embedding model path: {embedding_model}")
        self.embedding_serving = LocalEmbeddingServing(
            model_name=embedding_model,
            device=args.device,
            max_workers=1,
        )

        self.database_manager = DatabaseManager(
            db_type="sqlite",
            config={"root_path": db_root},
        )
        self.logger.info(f"discovered databases: {self.database_manager.list_databases()}")

        self.storage = FileStorage(
            first_entry_file_name=input_path,
            cache_path=self.cache_dir,
            file_name_prefix=args.prefix,
            cache_type="jsonl",
        )

        # ---- step1: 可执行性过滤（EXPLAIN，不真正执行） ----
        self.sql_executability_filter_step1 = SQLExecutabilityFilter(
            database_manager=self.database_manager
        )
        # ---- step2: SQL 变体生成 ----
        self.sql_variation_generator_step2 = SQLVariationGenerator(
            llm_serving=self.llm_serving,
            database_manager=self.database_manager,
            num_variations=args.num_variations,
            prompt_template=SQLVariationGeneratorPrompt(),
        )
        # ---- step3: 变体可执行性过滤 ----
        self.sql_executability_filter_step3 = SQLExecutabilityFilter(
            database_manager=self.database_manager
        )
        # ---- step4: 问题生成（多候选 + embedding 选最优） ----
        self.text2sql_question_generator_step4 = Text2SQLQuestionGenerator(
            llm_serving=self.llm_serving,
            embedding_serving=self.embedding_serving,
            database_manager=self.database_manager,
            question_candidates_num=args.question_candidates,
            prompt_template=Text2SQLQuestionGeneratorPrompt(),
        )
        # ---- step5: 问题-SQL 一致性过滤 ----
        self.text2sql_correspondence_filter_step5 = Text2SQLCorrespondenceFilter(
            llm_serving=self.llm_serving,
            database_manager=self.database_manager,
            prompt_template=Text2SQLCorrespondenceFilterPrompt(),
        )
        # ---- step6: 训练 prompt 生成（零 LLM 调用） ----
        self.text2sql_prompt_generator_step6 = Text2SQLPromptGenerator(
            database_manager=self.database_manager,
            prompt_template=Text2SQLPromptGeneratorPrompt(),
        )
        # ---- step7: CoT 推理轨迹采样 ----
        self.sql_cot_generator_step7 = Text2SQLCoTGenerator(
            llm_serving=self.llm_serving,
            database_manager=self.database_manager,
            prompt_template=Text2SQLCotGeneratorPrompt(),
            sampling_num=args.cot_sampling,
        )
        # ---- step8: CoT 执行一致性投票 ----
        self.sql_cot_voting_generator_step8 = Text2SQLCoTVotingGenerator(
            database_manager=self.database_manager
        )
        # ---- step9: SQL 组件难度分级（纯规则解析） ----
        self.sql_component_classifier_step9 = SQLComponentClassifier(
            difficulty_thresholds=[2, 4, 6],
            difficulty_labels=["easy", "medium", "hard", "extra"],
        )
        # ---- step10: SQL 执行难度分级（多次生成 + 执行比对） ----
        # 原版 num_generations=10, thresholds=[2,5,9]；时间预算 + 20 RPM 限流下改为 3 次。
        # 注意算子内部限制：num_generations 必须 > 最后一个 threshold，
        # 否则会被自动放大到 ((last_threshold//5)+1)*5（最少 5 次），无法压到 3 次。
        # 因此 thresholds=[1,2]：3 次生成中 0~1 次结果一致=extra，2 次=hard，3 次=easy。
        self.sql_execution_classifier_step10 = SQLExecutionClassifier(
            llm_serving=self.llm_serving,
            database_manager=self.database_manager,
            num_generations=args.exec_generations,
            difficulty_thresholds=[1, 2],
            difficulty_labels=["extra", "hard", "easy"],
        )

    # ---------------- 统计辅助 ----------------
    def _snapshot(self, step: int):
        path = os.path.join(self.cache_dir, f"{self.args.prefix}_step{step}.jsonl")
        record = {"step": step, "file": os.path.basename(path)}
        try:
            df = pd.read_json(path, lines=True)
        except Exception as exc:  # noqa: BLE001
            record.update({"rows": 0, "read_error": str(exc)})
            self.stats.append(record)
            self.logger.warning(f"[STEP {step}] 读取 {path} 失败: {exc}")
            return None
        record["rows"] = int(len(df))
        record["columns"] = list(df.columns)
        record["non_null"] = {
            k: int(df[k].notna().sum()) for k in KEY_FIELDS if k in df.columns
        }
        self.stats.append(record)
        self.logger.info(
            f"[STEP {step}] rows={len(df)} cols={list(df.columns)}"
        )
        return df

    def forward(self):
        sql_key = "SQL"
        db_id_key = "db_id"
        question_key = "question"
        evidence_key = "evidence"

        self.sql_executability_filter_step1.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key,
        )
        self._snapshot(1)

        self.sql_variation_generator_step2.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key,
            output_sql_variation_type_key="sql_variation_type",
        )
        self._snapshot(2)

        self.sql_executability_filter_step3.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key,
        )
        self._snapshot(3)

        self.text2sql_question_generator_step4.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key,
            output_question_key=question_key,
            output_evidence_key=evidence_key,
        )
        self._snapshot(4)

        self.text2sql_correspondence_filter_step5.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key,
            input_question_key=question_key,
            input_evidence_key=evidence_key,
        )
        self._snapshot(5)

        self.text2sql_prompt_generator_step6.run(
            storage=self.storage.step(),
            input_question_key=question_key,
            input_db_id_key=db_id_key,
            input_evidence_key=evidence_key,
            output_prompt_key="prompt",
        )
        self._snapshot(6)

        self.sql_cot_generator_step7.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_question_key=question_key,
            input_db_id_key=db_id_key,
            input_evidence_key=evidence_key,
            output_cot_key="cot_reasoning",
        )
        self._snapshot(7)

        self.sql_cot_voting_generator_step8.run(
            storage=self.storage.step(),
            input_cot_responses_key="cot_responses",
            input_db_id_key=db_id_key,
            output_cot_key="cot_reasoning",
        )
        self._snapshot(8)

        self.sql_component_classifier_step9.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            output_difficulty_key="sql_component_difficulty",
        )
        self._snapshot(9)

        self.sql_execution_classifier_step10.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key,
            input_prompt_key="prompt",
            output_difficulty_key="sql_execution_difficulty",
        )
        self._snapshot(10)

        stats_path = os.path.join(self.cache_dir, "stats_refine.json")
        with open(stats_path, "w", encoding="utf-8") as fout:
            json.dump(
                {
                    "input": self.input_path,
                    "db_root": os.path.abspath(self.args.db_root),
                    "cache_dir": self.cache_dir,
                    "args": vars(self.args),
                    "steps": self.stats,
                },
                fout,
                ensure_ascii=False,
                indent=2,
            )
        self.logger.info(f"stats written to {stats_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入 jsonl（pipeline_refine 子集）")
    parser.add_argument("--cache-dir", default="./cache_refine")
    parser.add_argument("--prefix", default="task4_refine")
    parser.add_argument(
        "--db-root",
        default=os.environ.get("TEXT2SQL_DB_ROOT", DEFAULT_DB_ROOT),
    )
    parser.add_argument("--num-variations", type=int, default=2)
    parser.add_argument("--question-candidates", type=int, default=2)
    parser.add_argument("--cot-sampling", type=int, default=2)
    parser.add_argument("--exec-generations", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-model", default="")
    return parser.parse_args()


if __name__ == "__main__":
    pipeline = Text2SQLRefineTask4(parse_args())
    pipeline.forward()
