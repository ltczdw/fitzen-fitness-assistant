# FitZen 健身助手

基于 FastAPI、Vue 3 / TypeScript 与 PostgreSQL / pgvector 的本地健身记录与问答项目。

- 训练记录：动作、逐组重量与次数、日期、编辑删除及统计。
- 饮食记录：食物查表、份量计算与营养汇总。
- AI对话：ReAct工具调用、知识检索、多轮计划修改及受预算约束的短期记忆。
- 保存与更正：原文授权、事务、幂等回执；失败不会冒充保存成功。

当前为单用户本地原型，尚无身份认证，不适合直接开放公网。健康回答用于一般信息参考，不能替代专业诊断。

## 安装与启动

环境：Python 3.12、Node 24、PostgreSQL 16+ 与 pgvector。Redis和模型服务按需配置。

Windows PowerShell，在仓库根目录：

```powershell
py -3.12 -m venv backend/.venv
backend/.venv/Scripts/python.exe -m pip install -r backend/requirements.txt
backend/.venv/Scripts/python.exe -m pip install --no-deps --no-build-isolation -e backend
backend/.venv/Scripts/python.exe -m pip check
npm ci --prefix frontend
```

数据库、配置示例、Ubuntu命令和启动步骤见[安装说明](docs/getting-started.md)。配置自己的数据库与供应商凭证；仓库仅提供 `.env.example`，不含真实配置或个人数据。

## 验证

```powershell
backend/.venv/Scripts/python.exe scripts/check_project.py --smoke
backend/.venv/Scripts/python.exe scripts/check_project.py
backend/.venv/Scripts/python.exe scripts/export_release.py --check
```

常规测试使用模型mock和临时SQLite；PostgreSQL测试需要显式隔离端点及创建/删除临时库权限，条件跳过不算通过。GitHub Actions执行核心回归、前端检查与发布清单校验；这些检查不等于真实模型质量评测。

训练安排未指定器械时默认健身房，明确居家、徒手或器械限制优先。默认不给时间表，也不追问时长；用户需要时间安排或提供限制时再处理。保存结果以实际数据库记录和回执为准。

## 文件与许可

源码、回归测试、合成评测输入、公开知识种子、依赖锁文件、CI和必要静态资源保留。历史报告、开发过程文档、会话管理工具及内部协作材料不包含在此副本中。

`frontend/design-system.md` 仅保留组件接口契约，`frontend/tokens.md` 是自动生成的Token清单；它们是前端构建检查的输入。

项目采用[MIT许可证](LICENSE)，第三方字体、知识资料与离线词表按各自许可证使用，完整归属见[第三方声明](THIRD_PARTY_NOTICES.md)。
