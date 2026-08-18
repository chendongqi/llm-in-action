"""
独立运行 generate_text_embeddings 步骤。

前置条件：output/ 下已有 entities.parquet、text_units.parquet、community_reports.parquet
效果：填充 lancedb/ 向量库（local search 所需）

用法：
  conda activate llm_base
  cd kb-03-graphrag-eval
  python build_embeddings.py
"""
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR / ".env")

WORKSPACE_DIR = SCRIPT_DIR / "data/workspace"

# 设置环境变量供 settings.yaml 的 ${VAR} 引用
for key in ["GRAPHRAG_LLM_API_KEY", "GRAPHRAG_LLM_BASE_URL",
            "GRAPHRAG_EMBEDDING_API_KEY", "GRAPHRAG_EMBEDDING_BASE_URL"]:
    os.environ.setdefault(key, os.environ.get(key, ""))


async def main():
    from graphrag.config.load_config import load_config
    from graphrag.index.run.utils import create_run_context
    from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
    # 直接从文件导入，避免触发 __init__.py 中的 spacy/torch 导入
    import importlib.util, sys
    _spec = importlib.util.spec_from_file_location(
        "generate_text_embeddings",
        "/home/ecarx/anaconda3/envs/llm_base/lib/python3.12/site-packages/"
        "graphrag/index/workflows/generate_text_embeddings.py"
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["graphrag.index.workflows.generate_text_embeddings"] = _mod
    _spec.loader.exec_module(_mod)
    generate_text_embeddings = _mod.generate_text_embeddings
    from graphrag_llm.embedding import create_embedding
    from graphrag.cache.cache_key_creator import cache_key_creator
    from graphrag_storage import create_storage, StorageConfig, StorageType
    from graphrag_storage.tables.parquet_table_provider import ParquetTableProvider
    from graphrag_cache import create_cache, CacheConfig, CacheType

    print(f"== 加载配置: {WORKSPACE_DIR} ==")
    config = load_config(root_dir=WORKSPACE_DIR)

    print("== 初始化存储和缓存 ==")
    output_storage = create_storage(
        StorageConfig(type=StorageType.File, base_dir=str(WORKSPACE_DIR / "output"))
    )
    cache_storage = create_storage(
        StorageConfig(type=StorageType.File, base_dir=str(WORKSPACE_DIR / "cache"))
    )
    output_table_provider = ParquetTableProvider(storage=output_storage)
    cache = create_cache(CacheConfig(type=CacheType.Json), storage=cache_storage)

    context = create_run_context(
        output_storage=output_storage,
        output_table_provider=output_table_provider,
        cache=cache,
        callbacks=NoopWorkflowCallbacks(),
    )

    # 验证所需 parquet 存在
    for table in ["entities", "text_units", "community_reports"]:
        exists = await context.output_table_provider.has(table)
        print(f"  {table}: {'✓' if exists else '✗ 缺失'}")

    print("== 初始化 Embedding 模型 ==")
    model_config = config.get_embedding_model_config(
        config.embed_text.embedding_model_id
    )
    model = create_embedding(
        model_config,
        cache=context.cache.child(config.embed_text.model_instance_name),
        cache_key_creator=cache_key_creator,
    )
    tokenizer = model.tokenizer

    print(f"== 开始生成 embeddings ==")
    print(f"  fields: {config.embed_text.names}")
    print(f"  向量库: {WORKSPACE_DIR}/lancedb")

    await generate_text_embeddings(
        config=config,
        table_provider=context.output_table_provider,
        callbacks=NoopWorkflowCallbacks(),
        model=model,
        tokenizer=tokenizer,
    )

    print("== embeddings 生成完毕 ==")

    lancedb_dir = WORKSPACE_DIR / "lancedb"
    tables = list(lancedb_dir.rglob("*.lance")) if lancedb_dir.exists() else []
    print(f"  LanceDB 表文件: {len(tables)} 个")
    for t in tables:
        print(f"    {t.relative_to(WORKSPACE_DIR)}")


if __name__ == "__main__":
    asyncio.run(main())
