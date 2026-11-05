import {readFile, readdir} from 'node:fs/promises'
import path from 'node:path'
import {fileURLToPath} from 'node:url'
import {createRequire} from 'node:module'
import ts from 'typescript'
import {frontendRoot, exportTokens, readTokens} from './export-tokens.mjs'
import {uiComponents, pageContainers, approvedVueFiles, tokenSource} from './design-system-policy.mjs'

const require = createRequire(import.meta.url)
const {parse:parseSfc} = require('vue/compiler-sfc')
const {parse:parseTemplate} = require('@vue/compiler-dom')
const postcss = require('postcss')
const themeColors = require('../tailwind.config.cjs').theme.extend.colors
// CSS names, including uncommon names and aliases. Keywords like transparent/currentColor are structural.
const namedColors = new Set(('aliceblue antiquewhite aqua aquamarine azure beige bisque black blanchedalmond blue blueviolet brown burlywood cadetblue chartreuse chocolate coral cornflowerblue cornsilk crimson cyan darkblue darkcyan darkgoldenrod darkgray darkgreen darkgrey darkkhaki darkmagenta darkolivegreen darkorange darkorchid darkred darksalmon darkseagreen darkslateblue darkslategray darkslategrey darkturquoise darkviolet deeppink deepskyblue dimgray dimgrey dodgerblue firebrick floralwhite forestgreen fuchsia gainsboro ghostwhite gold goldenrod gray green greenyellow grey honeydew hotpink indianred indigo ivory khaki lavender lavenderblush lawngreen lemonchiffon lightblue lightcoral lightcyan lightgoldenrodyellow lightgray lightgreen lightgrey lightpink lightsalmon lightseagreen lightskyblue lightslategray lightslategrey lightsteelblue lightyellow lime limegreen linen magenta maroon mediumaquamarine mediumblue mediumorchid mediumpurple mediumseagreen mediumslateblue mediumspringgreen mediumturquoise mediumvioletred midnightblue mintcream mistyrose moccasin navajowhite navy oldlace olive olivedrab orange orangered orchid palegoldenrod palegreen paleturquoise palevioletred papayawhip peachpuff peru pink plum powderblue purple rebeccapurple red rosybrown royalblue saddlebrown salmon sandybrown seagreen seashell sienna silver skyblue slateblue slategray slategrey snow springgreen steelblue tan teal thistle tomato turquoise violet wheat white whitesmoke yellow yellowgreen').split(' '))
const paletteNames = new Set('slate gray zinc neutral stone red orange amber yellow lime green emerald teal cyan sky blue indigo violet purple fuchsia pink rose black white'.split(' '))
const structuralColors = new Set(['transparent','current','currentColor','inherit','none','initial','unset','revert','revert-layer'])
const colorFunction = /\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color)\s*\(/gi

export function colorIssues(value, {names=true, tokens}={}) {
  const issues = []
  const text = String(value).replace(/\/\*[\s\S]*?\*\//g,'')
  if (/#(?:[\da-f]{8}|[\da-f]{6}|[\da-f]{4}|[\da-f]{3})(?![\w-])/i.test(text)) issues.push('硬编码 HEX 颜色')
  for (const match of text.matchAll(colorFunction)) {
    const rest = text.slice(match.index+match[0].length)
    if (!/^\s*var\(\s*--fz-rgb-[\w-]+\s*\)/i.test(rest)) issues.push('硬编码颜色函数；使用 rgb(var(--fz-rgb-*) / alpha)')
  }
  if (names) {
    // Strings and var() identifiers are not CSS color keywords.
    const stripped = text.replace(/(['"])(?:\\.|(?!\1).)*?\1/g,'').replace(/var\(\s*--[\w-]+/g,'var(')
    for (const word of stripped.match(/\b[a-z]+\b/gi)||[]) if (namedColors.has(word.toLowerCase())) issues.push(`硬编码命名颜色 ${word}`)
  }
  if (tokens) for (const match of text.matchAll(/var\(\s*(--[\w-]+)/g)) {
    if (!tokens.has(match[1])) issues.push(`未登记 Token ${match[1]}`)
  }
  return [...new Set(issues)]
}

export function classIssues(value) {
  const issues = []
  for (const token of String(value).split(/[\s'"`,]+/)) {
    const base = token.replace(/^(?:(?:[\w/-]+|\[[^\]]+\]):)+/,'').replace(/^!/, '')
    const match = base.match(/^(?:bg|text|border(?:-[trblxy])?|ring(?:-offset)?|outline|divide|fill|stroke|from|via|to|decoration|caret|accent|shadow|placeholder)-(.+)$/)
    if (!match) continue
    const role = match[1].split('/')[0]
    if (paletteNames.has(role.split('-')[0]) || namedColors.has(role.toLowerCase()) || colorIssues(role,{names:false}).length || /^\[(?:[a-z-]+:)?(?:white|black|red|green|blue)\]$/i.test(role)) issues.push(`未登记颜色类 ${token}`)
    // Numeric/font/layout arbitrary values are legal; semantic colors must be registered.
    if (/^[a-z]/i.test(role) && !structuralColors.has(role) && !themeColors[role] && !/^(?:xs|sm|base|lg|xl|\d+xl|body-|label-|headline-|display-|metric-|code-|left|right|center|justify|start|end|ellipsis|clip|solid|dashed|dotted|double|hidden|none|collapse|separate|offset-|inset|normal|medium|semibold|bold|italic|gradient-|cover|contain|auto|fixed|local|scroll|repeat|no-repeat)/.test(role) && /^(?:bg|fill|stroke|from|via|to|decoration|caret|accent)-/.test(base)) issues.push(`未登记颜色角色 ${role}`)
  }
  return [...new Set(issues)]
}

export function inventoryIssues(files) {
  const current = files.filter(file=>/\.(vue|jsx|tsx)$/i.test(file))
  return [
    ...current.filter(file=>!approvedVueFiles.includes(file)).map(file=>`禁止新增 / 复制组件：${file}`),
    ...approvedVueFiles.filter(file=>!current.includes(file)).map(file=>`登记组件 / 页面缺失：${file}`),
  ]
}

export function componentImportIssue(file, target) {
  if (target.startsWith('src/components/ui/')) return uiComponents.some(name=>target===`src/components/ui/${name}.vue`)?null:'未登记 UI 组件'
  if (target==='src/fitzen/App.vue' && file==='src/main.ts') return null
  if (pageContainers.includes(target) && file==='src/fitzen/App.vue') return null
  return '组件只能来自 src/components/ui；页面只能由 App 加载'
}

function scriptIssues(source, file, tokens, expression=false) {
  const ast = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
  const issues = []
  function visit(node) {
    if (ts.isImportDeclaration(node) && node.moduleSpecifier && !node.moduleSpecifier.text.startsWith('.') && node.moduleSpecifier.text!=='vue') {
      if (/\.vue$|(?:^|\/)(?:components|ui)(?:\/|$)|(?:element-plus|ant-design-vue|vuetify|primevue|naive-ui|radix-vue|reka-ui|@headlessui\/vue)/.test(node.moduleSpecifier.text)) issues.push(`禁止外部 UI 导入 ${node.moduleSpecifier.text}`)
    }
    const imported = ts.isImportDeclaration(node) || ts.isExportDeclaration(node) ? node.moduleSpecifier : ts.isCallExpression(node) && node.expression.kind===ts.SyntaxKind.ImportKeyword ? node.arguments[0] : null
    if (imported && ts.isStringLiteral(imported) && /\.vue$/.test(imported.text)) {
      const target = path.posix.normalize(path.posix.join(path.posix.dirname(file),imported.text))
      const issue = componentImportIssue(file,target)
      if (issue) issues.push(`${issue}：${imported.text}`)
    }
    if (ts.isCallExpression(node) && /(?:^|\.)(?:defineComponent|extend)$/.test(node.expression.getText(ast))) issues.push('禁止通过组件工厂新建组件')
    if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) {
      // Treat only entire color literals as colors; selectors and user text are not colors.
      if (/^\s*(?:#[\da-f]{3,8}|(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color)\s*\()/i.test(node.text)) issues.push(...colorIssues(node.text,{tokens}))
      issues.push(...classIssues(node.text))
      if (expression) issues.push(...colorIssues(node.text,{tokens}))
      const parent=node.parent
      const colorName = /(?:color|background(?:Color)?|fill|stroke|boxShadow)$/i
      const colorContext = (ts.isPropertyAssignment(parent) || ts.isVariableDeclaration(parent)) && colorName.test(parent.name.getText(ast)) || ts.isBinaryExpression(parent) && colorName.test(parent.left.getText(ast))
      if (colorContext) issues.push(...colorIssues(node.text,{tokens}))
    }
    ts.forEachChild(node,visit)
  }
  visit(ast)
  return [...new Set(issues)]
}

export function inspectSource(source, file, tokens) {
  const issues=[]
  function cssIssues(css) {
    try {
      postcss.parse(css).walkDecls(decl=>{
        if (!/^(?:content|font-family|animation(?:-name)?|transition(?:-property)?)$/i.test(decl.prop)) issues.push(...colorIssues(decl.value,{tokens}).map(issue=>`${decl.prop}: ${issue}`))
      })
    } catch(error) { issues.push(`CSS 解析失败：${error.message}`) }
  }
  if (file.endsWith('.vue')) {
    const {descriptor,errors}=parseSfc(source,{filename:file})
    issues.push(...errors.map(error=>`Vue 解析失败：${String(error)}`))
    for (const block of [descriptor.script,descriptor.scriptSetup].filter(Boolean)) issues.push(...scriptIssues(block.content,file,tokens))
    descriptor.styles.forEach(style=>cssIssues(style.content))
    if (descriptor.template) {
      const ast=parseTemplate(descriptor.template.content)
      function walk(node) {
        if (node.tag==='button' && file!=='src/components/ui/DeleteButton.vue') {
          const destructiveAttribute=(node.props||[]).some(prop=>prop.type===6 && (['data-delete-confirm','data-training-delete'].includes(prop.name) || ['aria-label','title'].includes(prop.name) && /删除|清空|清除上下文/.test(prop.value?.content||'')))
          const destructiveContent=/\bdelete(?:_forever|_sweep)?\b|>\s*(?:删除|清空全部|清除上下文)\s*</.test(node.loc.source)
          if (destructiveAttribute || destructiveContent) issues.push('删除 / 清空入口必须复用 DeleteButton，不能手写按钮')
        }
        for (const prop of node.props||[]) {
          if (prop.type===6 && prop.value) {
            if (prop.name==='class') issues.push(...classIssues(prop.value.content))
            if (prop.name==='style') cssIssues(`a{${prop.value.content}}`)
            if (['fill','stroke','color','bgcolor','stop-color','flood-color','lighting-color'].includes(prop.name)) issues.push(...colorIssues(prop.value.content,{tokens}))
          }
          if (prop.type===7 && prop.exp) {
            issues.push(...classIssues(prop.exp.content))
            if (prop.arg?.content==='style') issues.push(...scriptIssues(prop.exp.content,file,tokens,true))
            if (['fill','stroke','color','stop-color'].includes(prop.arg?.content)) issues.push(...scriptIssues(prop.exp.content,file,tokens,true))
          }
        }
        for (const child of node.children||[]) walk(child)
      }
      walk(ast)
    }
  } else if (file.endsWith('.css')) cssIssues(source)
  else issues.push(...scriptIssues(source,file,tokens))
  return [...new Set(issues)]
}

async function collect(directory) {
  const files=[]
  for (const entry of await readdir(path.join(frontendRoot,directory),{withFileTypes:true})) {
    const relative = `${directory}/${entry.name}`
    if (entry.isDirectory()) files.push(...await collect(relative))
    else files.push(relative)
  }
  return files
}

export async function checkDesignSystem() {
  const files=await collect('src')
  const issues=inventoryIssues(files)
  const tokens=readTokens(await readFile(path.join(frontendRoot,tokenSource),'utf8'))
  for (const [name,value] of tokens) {
    if (/^#[\da-f]{6}$/i.test(value)) {
      const rgbName=name.replace('--fz-','--fz-rgb-')
      const rgb=value.slice(1).match(/../g).map(channel=>Number.parseInt(channel,16)).join(' ')
      if (tokens.get(rgbName)!==rgb) issues.push(`${name} 的 RGB 配对不一致 / 缺失：应为 ${rgb}`)
    }
    if (value.includes('var(')) issues.push(...colorIssues(value,{tokens}).map(issue=>`${name}: ${issue}`))
  }
  for (const [name,value] of Object.entries(themeColors)) issues.push(...colorIssues(value,{tokens}).map(issue=>`tailwind.colors.${name}: ${issue}`))
  for (const file of files.filter(file=>/\.(vue|css|ts|js|tsx|jsx)$/.test(file) && file!==tokenSource)) {
    issues.push(...inspectSource(await readFile(path.join(frontendRoot,file),'utf8'),file,tokens).map(issue=>`${file}: ${issue}`))
  }
  const index=await readFile(path.join(frontendRoot,'src/components/ui/index.ts'),'utf8')
  const exported=[...index.matchAll(/default\s+as\s+(\w+)/g)].map(([,name])=>name).sort()
  if (JSON.stringify(exported)!==JSON.stringify([...uiComponents].sort())) issues.push('UI 导出入口与冻结清单不一致')
  for (const name of uiComponents) if (!new RegExp(`export\\s*\\{\\s*default\\s+as\\s+${name}\\s*\\}\\s*from\\s*['"]\\./${name}\\.vue['"]`).test(index)) issues.push(`UI 导出必须指向登记实现：${name}`)
  const docs=await readFile(path.join(frontendRoot,'design-system.md'),'utf8').catch(()=>'')
  for (const name of uiComponents) if (!docs.includes(`### ${name}`)) issues.push(`design-system.md 缺少组件契约 ${name}`)
  try { await exportTokens({check:true}) } catch(error) { issues.push(error.message) }
  if (issues.length) throw new Error([...new Set(issues)].join('\n'))
  return {components:uiComponents.length,files:files.length,tokens:tokens.size}
}

if (process.argv[1] && path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  try {
    const result=await checkDesignSystem()
    console.log(`设计系统检查通过：${result.components} 个 UI 组件，${result.tokens} 个 CSS Token；组件边界、颜色与文档一致。`)
  } catch(error) { console.error(error.message); process.exitCode=1 }
}
