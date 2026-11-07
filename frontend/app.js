const API_URL = 'https://blog-agent-699e.onrender.com';
let selectedLength = 'medium';
let progressTimers = [];

// Wake up Render on page load
fetch(`${API_URL}/docs`).catch(() => {});

function selectLength(length) {
  selectedLength = length;
  document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
  document.querySelector(`[data-length="${length}"]`).classList.add('active');
}

function show(id) { document.getElementById(id).classList.remove('hidden'); }
function hide(id) { document.getElementById(id).classList.add('hidden'); }

function setStep(n, status) {
  const step = document.getElementById(`step${n}`);
  if (!step) return;
  step.className = 'step ' + status;
}

function clearProgressTimers() {
  progressTimers.forEach(clearTimeout);
  progressTimers = [];
}

function resetForm() {
  clearProgressTimers();
  hide('outputSection');
  hide('progressSection');
  show('formSection');
  document.getElementById('topicInput').value = '';
  document.getElementById('audienceInput').value = '';
  for (let i = 1; i <= 9; i++) setStep(i, '');
}

async function generateBlog() {
  const topic = document.getElementById('topicInput').value.trim();
  const audience = document.getElementById('audienceInput').value.trim();

  if (!topic) {
    document.getElementById('topicInput').focus();
    document.getElementById('topicInput').style.borderColor = '#ef4444';
    setTimeout(() => document.getElementById('topicInput').style.borderColor = '', 2000);
    return;
  }

  const btn = document.getElementById('generateBtn');
  btn.disabled = true;
  document.getElementById('btnText').textContent = 'Generating...';

  hide('formSection');
  show('progressSection');

  clearProgressTimers();
  const stepTimings = [0, 3000, 6000, 12000, 15000, 30000, 50000, 60000, 65000];
  stepTimings.forEach((delay, i) => {
    progressTimers.push(setTimeout(() => {
      if (i > 0) setStep(i, 'done');
      setStep(i + 1, 'active');
    }, delay));
  });

  let timeout;
  try {
    const controller = new AbortController();
    timeout = setTimeout(() => controller.abort(), 360000); // 6 min timeout

    const response = await fetch(`${API_URL}/generate-blog`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic, audience: audience || 'general readers', length: selectedLength }),
      signal: controller.signal
    });

    clearTimeout(timeout);
    clearProgressTimers();

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      const errMsg = errData.detail || 'Something went wrong. Please try again.';
      throw new Error(errMsg);
    }
    const data = await response.json();

    for (let i = 1; i <= 9; i++) setStep(i, 'done');

    setTimeout(() => {
      hide('progressSection');
      renderOutput(data);
      show('outputSection');
    }, 800);

  } catch (err) {
    clearTimeout(timeout);
    clearProgressTimers();
    hide('progressSection');
    show('formSection');
    btn.disabled = false;
    document.getElementById('btnText').textContent = 'Generate Blog';
    if (err.name === 'AbortError') {
      alert('Request timed out after 6 minutes. Check the backend logs before retrying.');
    } else {
      alert(err.message || 'Something went wrong. Please try again.');
    }
  }
}

function renderOutput(data) {
  renderScore(data.scores);
  renderBlog(data);
}

function renderScore(scores) {
  const overall = parseFloat(scores['Overall Score']) || 0;
  const dims = [
    { key: 'Readability', label: 'Read' },
    { key: 'Hook Strength', label: 'Hook' },
    { key: 'Content Depth', label: 'Depth' },
    { key: 'SEO Strength', label: 'SEO' },
    { key: 'Conclusion Quality', label: 'End' },
  ];

  const barsHtml = dims.map(d => {
    const val = parseFloat(scores[d.key]) || 0;
    const pct = (val / 10) * 100;
    return `
      <div class="score-bar-item">
        <div class="score-bar-track">
          <div class="score-bar-fill" style="height:${pct}%"></div>
        </div>
        <span class="score-bar-label">${d.label}</span>
      </div>`;
  }).join('');

  document.getElementById('scoreCard').innerHTML = `
    <div class="score-overall">
      <span class="score-number">${overall}</span>
      <span class="score-denom">/10</span>
    </div>
    <p class="score-verdict">${scores['Verdict'] || ''}</p>
    <div class="score-bars">${barsHtml}</div>
  `;
}

async function fetchUnsplashImage(query) {
  try {
    const res = await fetch(`https://source.unsplash.com/800x400/?${encodeURIComponent(query)}`);
    return res.url;
  } catch {
    return null;
  }
}

function cleanMarkdown(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/\*(.*?)\*/g, '$1')
    .replace(/^#+\s/gm, '')
    .trim();
}

// Render inline markdown: **bold**, *italic*, [text](url)
function renderInline(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}

async function renderBlog(data) {
  const { topic, final_blog, extras, seo } = data;

  // TL;DR
  const tldrItems = (extras.tldr || []).map(b => `<li>${b}</li>`).join('');
  const tldrHtml = tldrItems ? `
    <div class="blog-tldr">
      <div class="blog-tldr-title">TL;DR</div>
      <ul>${tldrItems}</ul>
    </div>` : '';

  // Meta
  const readTime = seo['Estimated Read Time'] || '';
  const metaHtml = `
    <div class="blog-meta">
      ${readTime ? `<span class="blog-meta-item">${readTime}</span>` : ''}
      <span class="blog-meta-item">${data.length || 'medium'}</span>
      <span class="blog-meta-item">AI Generated</span>
    </div>`;

  // Parse blog into sections and inject images
  const lines = final_blog.split('\n');
  let bodyHtml = '';
  let imageCount = 0;
  const maxImages = data.length === 'long' ? 3 : data.length === 'medium' ? 2 : 1;
  const imageKeywords = topic.split(' ').slice(0, 3).join(' ');
  let headingCount = 0;

  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];
    const line = raw.trim();
    if (!line) continue;

    // --- Explicit markdown heading detection ---
    const h1Match = line.match(/^#\s+(.+)/);
    const h2Match = line.match(/^##\s+(.+)/);
    const h3Match = line.match(/^###\s+(.+)/);

    if (h3Match) {
      bodyHtml += `<h3 class="blog-subsection-heading">${renderInline(h3Match[1])}</h3>`;
    } else if (h2Match) {
      headingCount++;
      bodyHtml += `<h2 class="blog-section-heading">${renderInline(h2Match[1])}</h2>`;
      // Inject image after every 2nd h2 heading
      if (imageCount < maxImages && headingCount % 2 === 0) {
        const imgUrl = `https://source.unsplash.com/800x400/?${encodeURIComponent(imageKeywords)}&sig=${imageCount}`;
        bodyHtml += `<img src="${imgUrl}" alt="${topic}" class="blog-image" loading="lazy" />`;
        imageCount++;
      }
    } else if (h1Match) {
      // H1 inside the body (e.g. subtitle line from LLM) — treat as H2
      headingCount++;
      bodyHtml += `<h2 class="blog-section-heading">${renderInline(h1Match[1])}</h2>`;
    } else if (line.startsWith('---') || line.startsWith('***')) {
      // Horizontal rule — skip, they're just separators in the LLM output
      continue;
    } else {
      bodyHtml += `<p>${renderInline(line)}</p>`;
    }
  }

  // Pull quote
  const pullQuoteHtml = extras.pull_quote ? `
    <blockquote class="blog-pull-quote">"${cleanMarkdown(extras.pull_quote)}"</blockquote>` : '';

  // Key takeaway
  const takeawayHtml = extras.key_takeaway ? `
    <div class="blog-key-takeaway">
      <div class="blog-key-takeaway-label">Key Takeaway</div>
      <p>${cleanMarkdown(extras.key_takeaway)}</p>
    </div>` : '';

  // SEO section
  const seoHtml = `
    <div class="seo-section">
      <div class="seo-title">SEO Metadata</div>
      <div class="seo-grid">
        <div class="seo-item"><label>Meta Title</label><p>${seo['Meta Title'] || ''}</p></div>
        <div class="seo-item"><label>Focus Keyword</label><p>${seo['Focus Keyword'] || ''}</p></div>
        <div class="seo-item"><label>Meta Description</label><p>${seo['Meta Description'] || ''}</p></div>
        <div class="seo-item"><label>Read Time</label><p>${seo['Estimated Read Time'] || ''}</p></div>
      </div>
    </div>`;

  document.getElementById('blogContent').innerHTML = `
    ${tldrHtml}
    ${metaHtml}
    <h1 class="blog-title">${topic}</h1>
    <div class="blog-body">${bodyHtml}</div>
    ${pullQuoteHtml}
    ${takeawayHtml}
    ${seoHtml}
  `;
}

function downloadPDF() {
  const topic = document.getElementById('topicInput').value.trim() || 'blog';
  const filename = topic.replace(/[^a-z0-9]/gi, '_').toLowerCase().slice(0, 50) + '.pdf';
  const element = document.getElementById('blogContent');

  // Inject dark-theme override styles so html2canvas captures them correctly
  const styleTag = document.createElement('style');
  styleTag.id = 'pdf-dark-styles';
  styleTag.textContent = `
    .blog-article, #blogContent { background: #111111 !important; color: #f0ece4 !important; }
    .blog-title { color: #f0ece4 !important; }
    .blog-section-heading { color: #f0ece4 !important; }
    .blog-subsection-heading { color: #c9a96e !important; }
    .blog-body { color: #d4cfc7 !important; }
    .blog-body p { color: #d4cfc7 !important; }
    .blog-tldr { background: #1a1a1a !important; border-left: 3px solid #c9a96e !important; }
    .blog-tldr-title { color: #c9a96e !important; }
    .blog-tldr li { color: #888880 !important; }
    .blog-meta-item { color: #555550 !important; background: #1a1a1a !important; border: 1px solid #333 !important; }
    .blog-pull-quote { color: #e8d5b0 !important; border-left: 3px solid #c9a96e !important; }
    .blog-key-takeaway { background: #1a1a1a !important; border: 1px solid rgba(201,169,110,0.3) !important; }
    .blog-key-takeaway-label { color: #c9a96e !important; }
    .blog-key-takeaway p { color: #f0ece4 !important; }
    .seo-section { background: #1a1a1a !important; border: 1px solid #333 !important; }
    .seo-title { color: #555550 !important; }
    .seo-item label { color: #555550 !important; }
    .seo-item p { color: #888880 !important; }
    /* Force page breaks before every section heading so html2canvas
       captures each "chunk" separately and jsPDF stitches them cleanly */
    .pdf-page-break { display: block; page-break-before: always; break-before: page; height: 0; }
  `;
  document.head.appendChild(styleTag);

  // Insert explicit page-break divs before every H2 (except the very first)
  // so html2pdf splits the canvas at natural section boundaries.
  const headings = element.querySelectorAll('h2.blog-section-heading');
  const injected = [];
  headings.forEach((h, idx) => {
    if (idx === 0) return; // don't break before the first heading
    const br = document.createElement('div');
    br.className = 'pdf-page-break';
    h.parentNode.insertBefore(br, h);
    injected.push(br);
  });

  const opt = {
    margin: [12, 12, 12, 12],
    filename,
    image: { type: 'jpeg', quality: 0.97 },
    html2canvas: {
      scale: 2,
      backgroundColor: '#111111',
      useCORS: true,
      logging: false,
      scrollX: 0,
      scrollY: -window.scrollY,
      windowWidth: document.documentElement.scrollWidth
    },
    jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' },
    pagebreak: {
      // 'css' reads our page-break-before rules on .pdf-page-break divs
      // 'avoid-all' prevents slicing inside any block element
      mode: ['css', 'avoid-all']
    }
  };

  html2pdf().set(opt).from(element).save().then(() => {
    // Clean up injected break divs
    injected.forEach(el => el.remove());
    const tag = document.getElementById('pdf-dark-styles');
    if (tag) tag.remove();
  });
}
