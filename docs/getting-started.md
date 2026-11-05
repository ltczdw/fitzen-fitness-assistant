# 安装与本地启动

使用发布清单导出的完整副本或包含清单内全部文件的 clone。Python3.12、Node24；PostgreSQL16+pgvector。Redis、独立MCP和模型均可选。发布副本不包含真实配置、个人记录、虚拟环境或node_modules。

## 安装依赖

从仓库根，Windows PowerShell：

```powershell
py -3.12 -m venv backend/.venv
backend/.venv/Scripts/python.exe -m pip install -r backend/requirements.txt
backend/.venv/Scripts/python.exe -m pip install --no-deps --no-build-isolation -e backend
backend/.venv/Scripts/python.exe -m pip check
npm ci --prefix frontend
backend/.venv/Scripts/python.exe scripts/check_project.py
```

Ubuntu，使用已安装的Python3.12及对应venv支持：

```bash
python3.12 -m venv backend/.venv
backend/.venv/bin/python -m pip install -r backend/requirements.txt
backend/.venv/bin/python -m pip install --no-deps --no-build-isolation -e backend
backend/.venv/bin/python -m pip check
npm ci --prefix frontend
backend/.venv/bin/python scripts/check_project.py
```

固定依赖的Windows专用轮子带 `sys_platform == "win32"` 条件。构建/安装工具和测试工具仍在requirements中，pyproject声明直接运行依赖；保留受测版本，不升级供应商SDK。HTML管理页面和离线Tokenizer词表、来源清单及许可证包含在Python包中。当前GitHub Actions配置为Windows核心回归与发布清单检查；Ubuntu命令是安装入口，不代表已在Ubuntu运行验证。隔离PostgreSQL验收另用下文的显式入口，不连接个人数据库。

## 配置本机数据库

已有非生产数据库可直接配置。使用Compose时，在根目录复制 `.env.example` 为 `.env`，为POSTGRES_PASSWORD设置自己的密码，再执行：

```bash
docker compose up -d postgres redis
docker compose exec postgres pg_isready -U fitness -d fitness
```

Docker只用于数据库/可选Redis；API和前端在主机运行。端口5432/6379绑定回环地址，已占用时不要启动第二实例。Windows用 `Copy-Item`，Ubuntu用 `cp`；真实 `.env` 均被忽略，不放入发布副本。

复制 `backend/.env.example` 为 `backend/.env`，填写匹配账号/密码/数据库的DATABASE_URL，连接协议为postgresql+psycopg，地址使用127.0.0.1。密码需URL编码。已有服务由管理员先安装pgvector扩展，应用使用已存在扩展。根 `.env` 只供Compose，不替代后端配置。

ReAct 默认开启：前端普通对话走 `react_readonly`，显式记录走 `react_write`，未声明 mode 的 API 调用由 `CHAT_DEFAULT_MODE=auto` 自动选择（写入且运行时就绪走 `react_write`，纯读走 `react_readonly`，其余 legacy）。需要整体回退时把 `CHAT_DEFAULT_MODE` 设为 `legacy`；无模型凭证的离线环境请显式设为 `REACT_READONLY_ENABLED=false`、`REACT_WRITE_ENABLED=false`，恢复 legacy 规则兜底；无需模型时保留所有供应商凭证为空。关闭不用的Redis/后台记忆/自动摘要，MCP_URL留空即可用本地工具。需要模型时由操作者自行配置LLM；embedding/rerank分别配置，缺少它们会有明确降级，不能把规则模板当真实模型效果。不要把密钥放进前端。

## 启动与停止

Windows根目录启动API：

```powershell
backend/.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8001
```

Ubuntu：

```bash
backend/.venv/bin/python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8001
```

另一个终端，从根目录启动前端：

```bash
npm --prefix frontend run dev -- --host 127.0.0.1
```

浏览器打开127.0.0.1:5173；后端health在127.0.0.1:8001/api/health。ready表示数据库已初始化且可连接；模型configured_unverified只表示配置，未验证供应商。首次启动会初始化应用表；本项目还没有正式版本迁移体系，现有数据库升级前需要备份与另行验证，不对已有记录做试验。

端口变化需同步前端VITE_API_TARGET和后端CORS_ORIGINS。两个终端Ctrl+C停止服务；关闭聊天写开关只阻止新的react_write，不代替停止整个API。Windows可使用 `scripts/start-local.ps1` 的单实例辅助入口；它不会安装依赖或替你生成真实配置，会输出自己启动的PID。只按已核对的PID停止自己的进程。

可选公开摘要导入：进入backend，以对应平台的虚拟环境Python执行 `-m scripts.ingest_official_knowledge`。它不导入个人训练/饮食记录。合成输入与模型mock的隔离联调是测试能力，不是开启真实聊天写入的授权。

## 模型上下文预算

非敏感的项目运行参数保存在 `backend/runtime-settings.json`，当前上下文窗口为65,536，输出预留4,096，安全余量15%，可用输入预算51,609。配置优先级为：显式构造参数 → 进程环境变量 → runtime-settings.json → .env → 默认值。这样已有本地密钥文件无需修改；需要部署覆盖时可设置进程环境变量 `LLM_CONTEXT_WINDOW_TOKENS`。重启后通过 `/api/health` 的 `context_budget` 核对实际值。该窗口不改变单任务调用/Token预算或Agent步数上限。

## 最近对话与按需历史

当前 Wiki 分支默认读取本会话最近10轮已完成、未隐藏/删除、同epoch的用户与助手对话，供路由理解补充、更正和指代。`runtime-settings.json` 固定 `memory_short_window_turns=10`、`memory_recent_max_tokens=12000`；进程变量分别为 `MEMORY_SHORT_WINDOW_TURNS`、`MEMORY_RECENT_MAX_TOKENS`。一轮是一次用户输入及其助手回复，不包括工具日志。显式设置轮数为0可运行原寻址专用对照模式。

近期内容按最新优先、完整问答装配；未知自然语言单元不做规则切句或任意截断。过大的轮次标为未装入，仍可经目录按需读取并沿现有语义单元机制处理。12,000是近期正文预算，来源标识等提示开销另由模型总输入预算约束；首轮也会按真实网关余量移除可选整轮，当前目标已选择的必要来源不能静默丢弃。更早历史继续使用原有读取/依赖闭包/独立预算。

生成任务只收到其选择的消息和必要条件，默认10轮不等于给每个worker全部10轮。历史原文不能授予保存权限，也不自动成为当前事实。已加载近期原文再次被Wiki选择时先核对来源版本，再复用且不重复扣读取成本；合并按消息ID去重，隐藏/删除和epoch变化仍会阻断引用。

重启后 `/api/health.memory.window_turns` 与 `recent_max_tokens` 可核对当前值；聊天响应 `metrics.recent_memory` 记录候选、装入、省略轮数和正文计数，省略不代表没有历史记录。计数器沿用 `context_budget.counter_mode`，没有配置模型分词器时使用保守UTF-8计数。此改动不扩大总窗口、模型调用次数、保存授权或安全边界。

