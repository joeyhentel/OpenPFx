The error is due to unescaped curly braces in your JavaScript inside an f-string. In Python f-strings, any `{` or `}` not part of a variable expression must be doubled (`{{` or `}}`).

For example, in your `st_html(f""" ... """)` section, you should change:
```python
(function(){
```
to:
```python
(function(){{
```
And similarly change all closing braces `}` to `}}`, except when they are part of Python’s `{variable}` expressions like `{idx}` or `{js_text}`.

So, this:
```python
btn.addEventListener('click', async () => {
    try {
        await navigator.clipboard.writeText(txt);
    } catch (e) {
        const ta = document.createElement('textarea');
        ta.value = txt; document.body.appendChild(ta); ta.select();
        try { document.execCommand('copy'); } catch(_) {}
        document.body.removeChild(ta);
    }
});
```
Becomes:
```python
btn.addEventListener('click', async () => {{
    try {{
        await navigator.clipboard.writeText(txt);
    }} catch (e) {{
        const ta = document.createElement('textarea');
        ta.value = txt; document.body.appendChild(ta); ta.select();
        try {{ document.execCommand('copy'); }} catch(_) {{}}
        document.body.removeChild(ta);
    }}
}});
```
Once you replace all non-variable braces with doubled braces, the SyntaxError will go away.
