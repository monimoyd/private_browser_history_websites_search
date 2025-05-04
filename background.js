// background.js
const EXCLUDED_DOMAINS = [
  'mail.google.com',
  'web.whatsapp.com',
  'facebook.com',
  'twitter.com'
];

// Store processed URLs to avoid duplicates
let processedUrls = new Set();

// Listen for tab updates
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url) {
    if (!isExcludedDomain(tab.url)) {
      processPage(tab.url, tabId);
    }
  }
});

function isExcludedDomain(url) {
  return EXCLUDED_DOMAINS.some(domain => url.includes(domain));
}

async function processPage(url, tabId) {
  if (processedUrls.has(url)) return;
  
  try {
    // Extract content from the page
    const content = await chrome.scripting.executeScript({
      target: { tabId },
      function: () => {
        // Remove script tags, style tags, and other unnecessary elements
        const clone = document.cloneNode(true);
        const scripts = clone.getElementsByTagName("script");
        const styles = clone.getElementsByTagName("style");
        while (scripts.length > 0) scripts[0].parentNode.removeChild(scripts[0]);
        while (styles.length > 0) styles[0].parentNode.removeChild(styles[0]);
        
        return {
          title: document.title,
          content: document.body.innerText,
          url: window.location.href
        };
      }
    });

    // Send to backend server for processing
    await sendToServer(content[0].result);
    processedUrls.add(url);
  } catch (error) {
    console.error('Error processing page:', error);
  }
}

async function sendToServer(data) {
  const response = await fetch('http://localhost:8001/process', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data)
  });
  return response.json();
}