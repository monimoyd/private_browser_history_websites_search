// content.js
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'highlight') {
    highlightText(request.text);
  }
});

function highlightText(text) {
  const range = document.createRange();
  const sel = window.getSelection();
  
  // Find the text in the page
  const walker = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_TEXT,
    null,
    false
  );

  let node;
  while (node = walker.nextNode()) {
    const pos = node.textContent.indexOf(text);
    if (pos >= 0) {
      range.setStart(node, pos);
      range.setEnd(node, pos + text.length);
      sel.removeAllRanges();
      sel.addRange(range);
      
      // Scroll the highlighted text into view
      range.startContainer.parentElement.scrollIntoView({
        behavior: 'smooth',
        block: 'center'
      });
      
      // Add visual highlight
      const span = document.createElement('span');
      span.className = 'chrome-extension-highlight';
      span.style.backgroundColor = 'yellow';
      range.surroundContents(span);
      break;
    }
  }
}