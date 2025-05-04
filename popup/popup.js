// popup.js
document.getElementById('searchButton').addEventListener('click', performSearch);
document.getElementById('searchInput').addEventListener('keypress', (e) => {
  if (e.key === 'Enter') performSearch();
});

async function performSearch() {
  const query = document.getElementById('searchInput').value;
  const resultsDiv = document.getElementById('results');
  resultsDiv.innerHTML = 'Searching...';
  
  try {
    const response = await fetch('http://localhost:8001/search', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query })
    });
    
    const results = await response.json();
    displayResults(results);
  } catch (error) {
    resultsDiv.innerHTML = 'Error performing search';
    console.error('Search error:', error);
  }
}

function displayResults(results) {
  const resultsDiv = document.getElementById('results');
  resultsDiv.innerHTML = '';
  
  results.forEach(result => {
    const resultElement = document.createElement('div');
    resultElement.className = 'result-item';
    resultElement.innerHTML = `
      <h3>${result.title}</h3>
      <p>${truncateText(result.content, 150)}</p>
      <a href="${result.url}" class="view-link">View Page</a>
    `;
    
    resultElement.querySelector('.view-link').addEventListener('click', (e) => {
      e.preventDefault();
      openAndHighlight(result.url, result.content);
    });
    
    resultsDiv.appendChild(resultElement);
  });
}

function truncateText(text, maxLength) {
  if (text.length <= maxLength) return text;
  return text.substr(0, maxLength) + '...';
}

function openAndHighlight(url, content) {
  chrome.tabs.create({ url }, (tab) => {
    // Wait for page to load before highlighting
    chrome.tabs.onUpdated.addListener(function listener(tabId, info) {
      if (tabId === tab.id && info.status === 'complete') {
        chrome.tabs.onUpdated.removeListener(listener);
        chrome.tabs.sendMessage(tab.id, {
          type: 'highlight',
          text: content
        });
      }
    });
  });
}