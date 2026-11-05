// Small, text-only Markdown subset. Every source character is escaped before
// generating our fixed HTML tags; source HTML and images are never interpreted.
type Block = {kind:'paragraph'|'heading'|'code'|'quote',text:string}
  | {kind:'list',items:string[],ordered:boolean,start:number}
  | {kind:'table',header:string[],rows:string[][]}

const escapeText=(text:string)=>text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;')
const inlineToken=/(`[^`\n]+`|\*\*[\s\S]+?\*\*|__[^\n]+?__|(?<!!)\[[^\]\n]+\]\([^\s)]+\))/g
function safeLink(value:string){
  try{const url=new URL(value);return ['https:','http:'].includes(url.protocol)&&!url.username&&!url.password?url.href:null}catch{return null}
}
function inline(text:string,html=true):string{
  let output='',last=0
  for(const match of text.matchAll(inlineToken)){
    output+=html?escapeText(text.slice(last,match.index)):text.slice(last,match.index)
    const token=match[0]
    if(token.startsWith('`')) output+=html?`<code>${escapeText(token.slice(1,-1))}</code>`:token.slice(1,-1)
    else if(token.startsWith('**')||token.startsWith('__')){
      const label=token.slice(2,-2).trim()
      output+=html?`<strong>${escapeText(label)}</strong>`:label
    }else{
      const parts=token.match(/^\[([^\]]+)\]\((.+)\)$/)!
      const href=safeLink(parts[2])
      output+=html?(href?`<a href="${escapeText(href)}" target="_blank" rel="noopener noreferrer">${escapeText(parts[1])}</a>`:escapeText(parts[1])):parts[1]+(href?`（${href}）`:'')
    }
    last=match.index!+token.length
  }
  return output+(html?escapeText(text.slice(last)):text.slice(last))
}
const heading=/^\s{0,3}#{1,6}\s+(.+?)(?:\s+#+)?\s*$/
const list=/^\s*(?:([-+*])\s+|(\d+)[.)、]\s+)(.+)$/
const separator=/^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/
function cells(line:string){
  return line.trim().replace(/^\|/,'').replace(/\|$/,'').split(/(?<!\\)\|/).map(cell=>cell.replace(/\\\|/g,'|').trim())
}
function blocks(text:string):Block[]{
  const lines=text.replace(/\r\n?/g,'\n').split('\n'),result:Block[]=[]
  let index=0
  const table=(i:number)=>lines[i]?.includes('|')&&separator.test(lines[i+1]||'')
  const special=(i:number)=>!lines[i].trim()||heading.test(lines[i])||list.test(lines[i])||/^\s*(`{3,}|~{3,})/.test(lines[i])||table(i)||/^\s*>/.test(lines[i])||/^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/.test(lines[i])
  while(index<lines.length){
    const line=lines[index]
    if(!line.trim()||/^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)){index++;continue}
    const fence=line.match(/^\s*(`{3,}|~{3,})/)
    if(fence){
      const content:string[]=[];index++
      const closing=new RegExp('^\\s*'+fence[1][0]+'{'+fence[1].length+',}\\s*$')
      while(index<lines.length&&!closing.test(lines[index]))content.push(lines[index++])
      if(index<lines.length)index++
      result.push({kind:'code',text:content.join('\n')});continue
    }
    const title=line.match(heading)
    if(title){result.push({kind:'heading',text:title[1]});index++;continue}
    if(table(index)){
      const header=cells(line),rows:string[][]=[];index+=2
      while(index<lines.length&&lines[index].trim()&&lines[index].includes('|'))rows.push(cells(lines[index++]))
      result.push({kind:'table',header,rows});continue
    }
    const item=line.match(list)
    if(item){
      const ordered=!!item[2],items:string[]=[]
      while(index<lines.length){
        const next=lines[index].match(list)
        if(!next||!!next[2]!==ordered)break
        items.push(next[3]);index++
      }
      result.push({kind:'list',items,ordered,start:ordered?Number(item[2]):1});continue
    }
    if(/^\s*>/.test(line)){
      const content:string[]=[]
      while(index<lines.length&&/^\s*>/.test(lines[index]))content.push(lines[index++].replace(/^\s*>\s?/,''))
      result.push({kind:'quote',text:content.join('\n')});continue
    }
    const content=[line];index++
    while(index<lines.length&&!special(index))content.push(lines[index++])
    result.push({kind:'paragraph',text:content.join('\n')})
  }
  return result
}

export function renderReply(text:string):string{
  return blocks(text).map(block=>{
    if(block.kind==='code')return `<pre><code>${escapeText(block.text)}</code></pre>`
    if(block.kind==='heading')return `<h3>${inline(block.text)}</h3>`
    if(block.kind==='quote')return `<blockquote>${inline(block.text)}</blockquote>`
    if(block.kind==='paragraph')return `<p>${inline(block.text)}</p>`
    if(block.kind==='list'){
      const tag=block.ordered?'ol':'ul'
      return `<${tag}${block.ordered?` start="${block.start}"`:''}>${block.items.map(item=>`<li>${inline(item)}</li>`).join('')}</${tag}>`
    }
    if(block.kind==='table'){
      const width=Math.max(block.header.length,...block.rows.map(row=>row.length))
      const row=(values:string[],tag:'th'|'td')=>'<tr>'+Array.from({length:width},(_,index)=>`<${tag}${tag==='th'?' scope="col"':''}>${inline(values[index]||'')}</${tag}>`).join('')+'</tr>'
      return `<div class="chat-answer-table" role="region" aria-label="回答表格" tabindex="0"><table><thead>${row(block.header,'th')}</thead><tbody>${block.rows.map(values=>row(values,'td')).join('')}</tbody></table></div>`
    }
    return ''
  }).join('')
}

export function replyPlainText(text:string):string{
  return blocks(text).map(block=>{
    if(block.kind==='code')return block.text
    if(block.kind==='list')return block.items.map((item,index)=>(block.ordered?`${block.start+index}. `:'')+inline(item,false)).join('\n')
    if(block.kind==='table')return [block.header,...block.rows].map(row=>row.map(cell=>inline(cell,false)).join('\t')).join('\n')
    return inline(block.text,false)
  }).join('\n\n')
}
