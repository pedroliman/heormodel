local function render_keywords(keywords)
  if not keywords or #keywords == 0 then
    return nil
  end

  local keywords_list = {}
  if type(keywords) == "table" then
    for _, keyword in ipairs(keywords) do
      if type(keyword) == "table" then
        table.insert(keywords_list, pandoc.utils.stringify(keyword))
      else
        table.insert(keywords_list, pandoc.utils.stringify(keyword))
      end
    end
  else
    return nil
  end

  if #keywords_list == 0 then
    return nil
  end

  local keywords_text = table.concat(keywords_list, ", ")

  return pandoc.Div(
    {
      pandoc.Para({
        pandoc.Strong("Keywords:"),
        pandoc.Space(),
        pandoc.Str(keywords_text)
      })
    },
    pandoc.Attr("", {"keywords-section"})
  )
end

return {
  {
    Meta = function(meta)
      if meta.keywords then
        if not meta._keywords_div then
          meta._keywords_div = true
        end
      end
    end
  },
  {
    Pandoc = function(doc)
      if doc.meta.keywords then
        local keywords_div = render_keywords(doc.meta.keywords)
        if keywords_div then
          table.insert(doc.blocks, keywords_div)
        end
      end
      return doc
    end
  }
}
