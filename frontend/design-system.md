# 组件接口契约

构建检查所需的现有组件接口与状态约束。颜色与尺寸见[Token清单](tokens.md)。

## 接口与状态契约

业务组件的接口以 `useFitzen()` / `ui` 状态和浏览器事件为主，以下为现有实现，不补造 `v-model`、`open`、`variant`、`size` 或 `@save` 接口。

### DeleteButton

| Prop | 类型 / 默认值 | 用途 |
| --- | --- | --- |
| `action` | `() => void \| Promise<unknown>`，必填 | 已确认后执行的业务操作；不含再次确认逻辑 |
| `subject` | `string`，必填 | 可访问名称里的实际删除对象，例“动作 卧推” |
| `label` | `string`，默认“删除” | 常态标签；清空草稿入口用“清空全部” |
| `disabled` | `boolean`，默认 `false` | 禁用；置为 true 时撤销待确认状态 |
| `loading` | `boolean`，默认 `false` | 外部忙碌状态；禁用并显示处理中 |
| `iconOnly` | `boolean`，默认 `false` | 窄操作列只显示图标，仍保留 aria-label / title |
| `compact` | `boolean`，默认 `false` | 宽屏显示标签，560px 以下只显示图标 |

- 红色垃圾桶与标签 → 首次点击切为绿色勾选 / “确认” → 2.5 秒内再次点击执行 `action`。同一时刻全局只允许一个按钮待确认。
- 超时、点击其他地方、输入、其他按键、组件卸载或禁用会撤销确认；在当前按钮使用 Enter / Space 可完成二次确认，按住按键不重复执行。
- 组件自动等待 action 的 Promise，期间禁用按钮。异常显示重试提示并发出 `error` 事件；具体 API 错误仍由业务操作反馈。
- 内置 `type="button"`、点击阻止冒泡、焦点轮廓、aria-label、title、aria-busy 与 `data-delete-confirm`。不得用外部颜色 / 尺寸类覆盖它；允许布局定位和既有自动化标记。
- 尺寸读取 `delete-height` / `delete-radius` / `delete-font-size` / `delete-icon-size`，颜色使用现有 `error`、`primary` 等 Token，无新色值。

```vue
<DeleteButton :action="()=>deleteAction(row)" :subject="'动作 '+row.name" :disabled="deleting" compact />
```

调用方只传原有删除请求或草稿修改，不再保留本地确认计时器。AI 对话页没有清除上下文入口；新建对话仍通过历史对话面板调用 `newChat`。组件本身不决定任何 URL，不删除数据。

### ChatMessages

- 从 `useFitzen()` 读取 `ui`，调用现有 `stopChat` / `retryChat` / `rateAnswer` / `confirmPending` / `resumeChat`，保留保存回执、待确认记录、过期状态、恢复原请求与回答反馈。无 `defineEmits`，无公开 ref 方法，无公开 slot。
- 在 `screens/Chat.vue` 的 `#chat-stream` 中挂载一次；滚动容器由页面提供。页面挂载及历史消息加载 / 切换完成后，默认滚动到最新消息；流式回复沿用既有滚动逻辑，仍可向上阅读历史消息。
- 回复操作保留复制、有帮助及未解决反馈，移除“质量与审核”入口。
- 保留空会话、流式回复、工具反馈与重试。回复下方不单独展示参考来源列表、内部来源标识或分块编号；引用元数据仍保留用于审核。正文外链只接受无用户名 / 密码的 HTTP(S) URL。
- AI正文由chat-format安全排版：支持段落、轻量标题、列表、强调、代码、表格与HTTP(S)链接，原文HTML不执行，不加载回答中的图片。流式正文与历史共用同一路径；用户消息保持原文。复制AI回答为可读纯文本，保留数量、单位和表格数据。模型提示默认短段落、少标题和少装饰；不改原始消息或保存回执。

### EnergyBalanceChart

```ts
// 当前实际 Props 签名是 { daily: any[] }；下方是调用方必须满足的数据约定。
type EnergyDay = {
  date: string          // YYYY-MM-DD，用于日期标签及列表 key
  calories: number | null
  tdee: number | null
}
```

- `daily` 按时间升序排列；无记录传 `[]`，单日无值用 `null`，真实零值用 `0`。不要省略字段传 `undefined`，当前逻辑只将 `null` 判为缺失。
- `calories` 与 `tdee` 单位均为 kcal；值应有限、非负，`date` 唯一。无 `defineEmits`，无公开 slot / ref 方法。
- 每日摄入和消耗分别使用 `chart-intake` / `chart-expenditure`；数据由调用方决定，组件不加载 API。

```vue
<script setup lang="ts">
import {EnergyBalanceChart} from '../../components/ui'
// 示例展示现有 screens 目录中的导入方式；数据仍来自既有业务状态。
</script>
<template><EnergyBalanceChart :daily="periodDays" /></template>
```

### FoodModal

- 打开新增饮食弹窗时即确定本餐时间（传入mealDate时使用该日加当前本地时分），空清单也允许修改；首个/后续食物、删空及清空不改时间，已有非空草稿重新打开保留时间，成功保存后清空时间。移除重复操作说明及副标题，保留字段/营养来源与AI估算事实、加载/空状态/错误/保存反馈；餐次与时间标签同基线、控件40px同高，窄屏纵向排列。编辑弹窗相同说明文字同步精简，不改已有记录日期。

- 在组合根 `App.vue` 挂载一次。由 `useFitzen().openFood(mealDate?)` 打开，内部观察 `ui.foodOpen`；不要另外维护一个不被组件读取的 `open` prop。
- 草稿、保存状态与错误来自 `ui`；内部调用现有食物 / 餐次 API。保存可能持久化用户数据，不能用真实保存做视觉测试。
- 组件负责 Teleport、焦点处理、Escape、取消与保存期间禁用关闭等行为。
- “食物库”统一展示、搜索内置食物和当前用户自定义食物，支持“自定义”分类；手工与 AI 保存沿用个人食物接口。选择个人食物保留 `source=custom` 及其营养，不以同名内置食物覆盖；打开库时刷新个人列表，按钮图标和文字居中对齐。
- 常用食物搜索在输入停顿250ms后展示内置与个人食物候选；回车/搜索按钮只执行只读查询，用户点击或键盘激活候选后才加入本餐。展示名称、来源/状态与每100g热量；无结果与失败可恢复，清空输入恢复推荐。请求组合取消信号与12秒超时，版本校验阻止过期结果覆盖，输入法确认不触发添加。
- 新增饮食弹窗的分类保留在已选食物清单下方、录入表单上方的原位置；中间内容可滚动，分类滚动到内容区上沿后吸顶，直到合计区域底部仍可切换。标题与底部操作固定，长内容不得被 flex 压缩裁切。

- 食物库内提供“新增食物”和“管理食物”：新增按每 100g 填写营养并独立保存到个人库，禁止提交当前餐次；管理模式暂停选择，内置与个人食物均使用 DeleteButton 二次确认删除，服务端以 is_active 停用并保留历史营养快照。失败保留条目/输入供重试，操作中阻止重复提交。两入口的库弹层 Teleport 至 body，独立滚动、焦点循环、Escape 分层关闭及焦点恢复；关闭/切换管理不得中断正在执行的写入。

- 管理模式条目提供“编辑”和统一删除。编辑复用每100g表单并回填名称/五项营养，按原来源与id通过PUT更新原条目；内置名称修改保留初始化身份，个人条目按user_id限定。保存成功更新库列表/缓存，不直接提交餐次或改写历史；重名409、已删除404、服务不可用503保留输入，取消不写库。保存时焦点留在顶层库弹层，避免按钮禁用后Esc触发外层关闭；表单错误展示于保存操作旁。管理最左、新增最右，分类全部/自定义固定前两项。

### NutritionEditModal

- 组合根挂载一次；当前 `App.vue` 的本地别名为 `NutritionRecordManager`，实际文件和导出名称仍是 `NutritionEditModal`。
- `fitness-edit-nutrition-record`：`CustomEvent` 的 `detail` 是完整 `RecordItem`，用于打开编辑。
- `fitness-open-nutrition-records`：打开现有饮食记录列表入口。成功保存 / 删除继续使用现有刷新逻辑，无 Vue `save` emit。
- 结构化食物记录与手工营养记录走不同编辑分支。保留并发估算、校验、确认删除、滚动锁定和焦点恢复。
- 添加食物使用与新增餐次相同的统一食物库；选择个人食物以自定义来源及营养提交，原有条目未改名时保持原始名称语义。
- 自定义表单及 AI 估算结果均在加入本餐操作旁提供独立的“保存到食物库”按钮；使用现有个人食物保存/校验与列表刷新，保留当前输入、AI结果和餐次草稿，type=button 避免提交餐次。AI结果保存成功后显示“已保存到食物库”，重新查询时重置保存状态。 新增和编辑两入口的自定义/AI操作统一为左“加入本餐”、右“保存到食物库”，两列等宽；同一入口两种表单复用相同按钮样式。
- 常用搜索沿用新增餐次的“只读查询 → 展示候选 → 明确选择加入”行为；搜索、回车和输入法确认不能提交编辑表单或自动添加。达到每餐上限时保留查询与结果供用户处理。

### RecordsDrawer

- 组合根挂载一次。`openRecords(tab?)` 打开；默认 `training`，饮食用 `nutrition`。显示与分页由 `ui.records`、`recordsTab`、`recordSearch`、`recordPage` 驱动。
- 具体编辑转交对应已有组件；含 `exercises` 的训练记录进入会话编辑器。删除仍走现有 API 和二次点击确认。
- 虽名为 Drawer，当前实现是记录管理覆盖层，不提供通用方向、尺寸、内容 slot 等 API。

### WorkoutSessionManager

- 训练录入提供“力量 / 有氧”胶囊分类；按动作库明确 category=有氧 归为有氧，其余部位及旧自定义类别归为力量，不按名称猜测。分类统一过滤常用推荐、搜索和动作库三页，力量保留部位筛选。切换仅改变候选，不清空已填写动作、逐组数据或搜索词；选中动作后同步分类，新增记录标题同步为力量训练/有氧训练，编辑保留原标题。自定义表单沿用所选大类；新开记录默认力量。有氧按距离（km）和时间（分钟）分段录入，首次选择有氧时未填写的时间默认1分钟，已有填写值保持，新增分段沿用上一段时间、无上一段时间时默认1；保存独立 distance_km/duration_minutes 字段；距离允许0，时间必须大于0，不允许空值、负值、非有限数或与重量次数混用。有氧只统计距离/时间，不参与力量负荷、完成力量组数或1RM；力量保留kg/次数。编辑按持久化指标回填，不依赖库名称，也不将旧重量/次数换算成距离/时间。切换候选不丢指标，真正换动作类型时分别保留草稿但只提交对应字段。


- 推荐区重新打开时保持缓存候选可见，后台刷新不将已有列表替换为loading；缓存读取失败仍保留候选和重试反馈。推荐/搜索/加载/空/错误共用稳定结果视口（桌面112px、手机160px），超出内容只在结果区滚动，避免下方逐组明细受异步数据高度推动；切换/清空搜索将结果滚动位置复位。外框不变并不代表内部不跳，验收须采样逐组标题y和推荐区域高度。

- 常用动作区提供独立搜索框与搜索按钮，按已加载动作库的名称/部位/肌群匹配，忽略大小写和空白、完全同名优先，结果上限20。输入停顿250ms或回车/搜索只更新候选，输入法组合期间不查询，点击候选才填写动作名并恢复常用推荐，保留日期和逐组数据。清空/Escape先清搜索，关闭/切换编辑清除待处理计时；本地过滤不增加写入或逐键请求，库读取错误可重试。库内搜索与表单搜索独立。

- 训练编辑器打开时外框即为height:min(860px,92dvh)、视口留白至少16px的稳定尺寸，头部不压缩，表单flex:1/min-height:0独立滚动、滚动条占位稳定且子项不被压缩。推荐加载/空/错误不改变外框位置或高度。首次聚焦及关闭恢复均preventScroll，去除同次打开的重复聚焦；只有editorOpen期间固定高度，保留原保存完成后的紧凑状态。

- 新增/编辑训练打开时只读加载动作库，表单直接显示服务端使用排序的前8个常用动作胶囊；点击仅填入当前动作名称并反馈选中，保留日期/组数/重量，手动保存才写训练。加载失败可重试，与完整动作库共用读取状态并合并并发请求，推荐不受库内搜索/分类筛选影响。动作库弹层固定宽度上限760px、高度min(760px,90dvh)且限制视口留白；标题、分类、搜索、底部固定，中间列表flex:1独立滚动并保留滚动条占位，加载/空结果/分类切换不改变外框尺寸。

- 动作名称/训练日期区域桌面双列、手机单列；标题行28px最小高度，输入外框44px等高并对齐，动作库入口在名称标题行右侧，保留输入内搜索入口。标签通过for/id关联。肌群与正值估算1RM独立换行，不显示空肌群占位或零1RM；周次/星期在日期下左右排列，长文字不撑出列宽。计算与保存script不随排版改动。

- 逐组体感反馈使用 Teleport 到 body 的 fixed listbox，按触发器位置和实际菜单高度选择上下展开并限制在视口内；避免组列表/表单 overflow 裁切，三个选项无需滚动即可看到。展开与选择使用 preventScroll 保持现有滚动位置，支持方向/Home/End/Enter、Tab 离开、Escape 仅收起菜单、外点关闭及触发器焦点恢复；滚动/resize 时更新位置，触发器离开可见表单区域则收起。沿用原三档映射与训练保存接口。

- 组合根挂载一次。现有 `openWorkout()` 派发 `fitness-open-workout-sessions`，打开新增会话流程。
- 编辑使用 `fitness-edit-workout-session` 的 `detail`：现有会话对象，含 `id`、`session_date`、`title`、`note`、`exercises`；每个动作含 `exercise_name` 和 `sets`。
- 每组使用 `weight_kg`、`reps`、`difficulty`、`completed`、`note` 等既有字段，不得将 UI 显示文本作为 API 字段名。
- 动作库保留推荐、完整目录与自定义动作；选择只填入草稿，保存动作库仍沿既有 API。
- 保存 / 删除后派发 `fitness-workout-records-changed`。这不是 Vue emit；监听器在挂载 / 卸载时成对注册 / 清理。
- 编辑表格使用 `fz-data-body` / `fz-data-row`；保存、删除、校验和键盘逻辑均保留。

### WorkoutRecordManager

- 组合根挂载一次。`fitness-open-workout-records` 打开管理界面；`fitness-edit-workout-record` 的 `detail` 是旧式记录。
- 旧式记录字段：`id`、`date`、`exercise`、`sets`、`reps`、`weight_kg`、`difficulty`、`completed`、`note`。
- 保存 / 删除后派发 `fitness-workout-records-changed`；无 Props、Vue emit、公开 slot 或 ref 方法。

现有调用方优先使用 `state.ts` 的 `openFood` / `openWorkout` / `openRecords` 和 `metrics.ts` 的 `editRecord`，不要在页面再手写一套事件分流。
