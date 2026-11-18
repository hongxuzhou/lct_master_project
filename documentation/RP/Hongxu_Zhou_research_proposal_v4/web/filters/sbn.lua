-- Keep the code block's text intact while adding presentation-only spans.
local function escape(text)
  return (text:gsub('&', '&amp;'):gsub('<', '&lt;'):gsub('>', '&gt;')
    :gsub('"', '&quot;'))
end

function CodeBlock(block)
  if not block.classes:includes('sbn') or not quarto.doc.is_format('html') then
    return nil
  end
  local lines = {}
  local highlight = block.attributes['highlight-lines']
  local first, last
  if highlight then
    first, last = highlight:match('^(%d+)%-(%d+)$')
    first, last = tonumber(first), tonumber(last)
  end
  for line in (block.text .. '\n'):gmatch('(.-)\n') do
    local content, comment = line:match('^(.-)(%%.*)$')
    content = content or line
    local indent, separator = content:match('^(%s*)(CORRECTION%s+[<>]%-?%d+%s*)$')
    if not separator then
      indent, separator = content:match('^(%s*)(CONJUNCTION%s+[<>]%-?%d+%s*)$')
    end
    if separator then
      -- Leave trailing alignment spaces outside the highlighted token.
      local token, trailing = separator:match('^(.-)(%s*)$')
      content = escape(indent) .. '<span class="sbn-separator">' .. escape(token)
        .. '</span>' .. escape(trailing)
    else
      content = escape(content)
    end
    local line_number = #lines + 1
    if first and line_number >= first and line_number <= last then
      local token, trailing = content:match('^(.-)(%s*)$')
      content = '<span class="sbn-error">' .. token .. '</span>' .. trailing
    end
    if comment then
      content = content .. '<span class="sbn-comment">' .. escape(comment) .. '</span>'
    end
    table.insert(lines, content)
  end
  return pandoc.RawBlock('html', '<div class="sbn-block"><pre class="sbn" tabindex="0" aria-label="SBN sequence"><code>'
    .. table.concat(lines, '\n') .. '</code></pre></div>')
end
