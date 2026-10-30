#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataFlow Task4 —— Text2SQL 数据合成（gen）流水线（最小改造版）

基于 open-dataflow 1.0.11 官方 api_pipelines/text2sql_pipeline_gen.py：
  无输入 SQL 样本，直接基于数据库 schema 生成 SQL（SQLGenerator），
  再走 可执行性过滤 -> 问题生成 -> 一致性过滤 -> prompt -> CoT -> 投票 -> 难度分级。
改动同 refine：AMD/DeepSeek-V4-Flash + 本地 bge embedding + 本地 SQLite 库 + 缩小生成量。
注意：官方用 example_data/Text2SQLPipeline/empty.jsonl 作为占位输入；
FileStorage 在 first_entry_file_name=="" 时直接返回空 DataFrame，这里用空字符串更干净。
"""
import argparse
import glob
import json
import os

import pandas as pd

from dataflow import get_logger
from dataflow.operators.text2sql import (
    SQLGenerator,
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
    SelectSQLGeneratorPrompt,
    Text2SQLCotGeneratorPrompt,
    Text2SQLQuestionGeneratorPrompt,
    Text2SQLPromptGeneratorPrompt,
    Text2SQLCorrespondenceFilterPrompt,
)
from dataflow.serving import APILLMServing_request, LocalEmbeddingServing
from dataflow.utils.storage import FileStorage
from dataflow.utils.text2sql.database_manager import DatabaseManager

DEFAULT_DB_ROOT = "/home/simin/dataflow-task4/db/downloaded_databases/databases"
DEFAULT_API_BASE = "https://developer.amd.com.cn/radeon/api/v1"
DEFAULT_MODEL = "DeepSeek-V4-Flash"

KEY_FIELDS = [
    "db_id", "SQL", "question", "evidence", "prompt", "cot_reasoning",
    "sql_complexity_type", "sql_component_difficulty", "sql_execution_difficulty",
]


def find_embedding_model() -> str:
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


class Text2SQLGenTask4:
    def __init__(self, args):
        self.args = args
        self.logger = get_logger()
        self.stats = []

        db_root = os.path.abspath(args.db_root)
        if not os.path.isdir(db_root):
            raise FileNotFoundError(f"数据库目录不存在: {db_root}")
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
            first_entry_file_name="",
            cache_path=self.cache_dir,
            file_name_prefix=args.prefix,
            cache_type="jsonl",
        )

        self.sql_generator_step1 = SQLGenerator(
            llm_serving=self.llm_serving,
            database_manager=self.database_manager,
            generate_num=args.generate_num,
            prompt_template=SelectSQLGeneratorPrompt(),
        )
        self.sql_executability_filter_step2 = SQLExecutabilityFilter(
            database_manager=self.database_manager
        )
        self.text2sql_question_generator_step3 = Text2SQLQuestionGenerator(
            llm_serving=self.llm_serving,
            embedding_serving=self.embedding_serving,
            database_manager=self.database_manager,
            question_candidates_num=args.question_candidates,
            prompt_template=Text2SQLQuestionGeneratorPrompt(),
        )
        self.text2sql_correspondence_filter_step4 = Text2SQLCorrespondenceFilter(
            llm_serving=self.llm_serving,
            database_manager=self.database_manager,
            prompt_template=Text2SQLCorrespondenceFilterPrompt(),
        )
        self.text2sql_prompt_generator_step5 = Text2SQLPromptGenerator(
            database_manager=self.database_manager,
            prompt_template=Text2SQLPromptGeneratorPrompt(),
        )
        self.sql_cot_generator_step6 = Text2SQLCoTGenerator(
            llm_serving=self.llm_serving,
            database_manager=self.database_manager,
            prompt_template=Text2SQLCotGeneratorPrompt(),
            sampling_num=args.cot_sampling,
        )
        self.sql_cot_voting_generator_step7 = Text2SQLCoTVotingGenerator(
            database_manager=self.database_manager
        )
        self.sql_component_classifier_step8 = SQLComponentClassifier(
            difficulty_thresholds=[2, 4, 6],
            difficulty_labels=["easy", "medium", "hard", "extra"],
        )
        self.sql_execution_classifier_step9 = SQLExecutionClassifier(
            llm_serving=self.llm_serving,
            database_manager=self.database_manager,
            num_generations=args.exec_generations,
            # 同 refine：num_generations 必须 > 最后 threshold，3 次生成用 [1,2]
            difficulty_thresholds=[1, 2],
            difficulty_labels=["extra", "hard", "easy"],
        )

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
        if "db_id" in df.columns and len(df) > 0:
            record["db_id_counts"] = {
                str(k): int(v) for k, v in df["db_id"].value_counts().items()
            }
        self.stats.append(record)
        self.logger.info(f"[STEP {step}] rows={len(df)} cols={list(df.columns)}")
        return df

    def forward(self):
        sql_key = "SQL"
        db_id_key = "db_id"
        question_key = "question"
        evidence_key = "evidence"

        self.sql_generator_step1.run(
            storage=self.storage.step(),
            output_sql_key=sql_key,
            output_db_id_key=db_id_key,
            output_sql_complexity_key="sql_complexity_type",
        )
        self._snapshot(1)

        self.sql_executability_filter_step2.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key,
        )
        self._snapshot(2)

        self.text2sql_question_generator_step3.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key,
            output_question_key=question_key,
            output_evidence_key=evidence_key,
        )
        self._snapshot(3)

        self.text2sql_correspondence_filter_step4.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key,
            input_question_key=question_key,
            input_evidence_key=evidence_key,
        )
        self._snapshot(4)

        self.text2sql_prompt_generator_step5.run(
            storage=self.storage.step(),
            input_question_key=question_key,
            input_db_id_key=db_id_key,
            input_evidence_key=evidence_key,
            output_prompt_key="prompt",
        )
        self._snapshot(5)

        self.sql_cot_generator_step6.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_question_key=question_key,
            input_db_id_key=db_id_key,
            input_evidence_key=evidence_key,
            output_cot_key="cot_reasoning",
        )
        self._snapshot(6)

        self.sql_cot_voting_generator_step7.run(
            storage=self.storage.step(),
            input_cot_responses_key="cot_responses",
            input_db_id_key=db_id_key,
            output_cot_key="cot_reasoning",
        )
        self._snapshot(7)

        self.sql_component_classifier_step8.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            output_difficulty_key="sql_component_difficulty",
        )
        self._snapshot(8)

        self.sql_execution_classifier_step9.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key,
            input_prompt_key="prompt",
            output_difficulty_key="sql_execution_difficulty",
        )
        self._snapshot(9)

        stats_path = os.path.join(self.cache_dir, "stats_gen.json")
        with open(stats_path, "w", encoding="utf-8") as fout:
            json.dump(
                {
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
    parser.add_argument("--cache-dir", default="./cache_gen")
    parser.add_argument("--prefix", default="task4_gen")
    parser.add_argument(
        "--db-root",
        default=os.environ.get("TEXT2SQL_DB_ROOT", DEFAULT_DB_ROOT),
    )
    parser.add_argument("--generate-num", type=int, default=2, help="每个数据库生成 SQL 条数")
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
    pipeline = Text2SQLGenTask4(parse_args())
    pipeline.forward()
