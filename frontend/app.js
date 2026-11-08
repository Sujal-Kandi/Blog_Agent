const API_URL = 'https://blog-agent-699e.onrender.com';
let selectedLength = 'medium';
let progressTimers = [];
let _lastTopic = ''; // store topic so PDF filename survives resetForm()

// Wake up Render on page load
fetch(`${API_URL}/docs`).catch(() => {});

// Re-enable generate button on load — guards against stuck state after refresh
window.addEventListener('load', () => {
  const btn = document.getElementById('generateBtn');
  if (btn) {
    btn.disabled = false;
    document.getElementById('btnText').textContent = 'Generate Blog';
  }
  hide('progressSection');
  hide('outputSection');
  show('formSection');
});

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

  _lastTopic = topic; // save before form gets cleared

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
  let h2Count = 0;
  let chapterCount = 0;
  let inSources = false;
  let sourcesHtml = '';
  let bodyLineCount = 0; // track how many body lines we've seen before allowing Sources detection

  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];
    const line = raw.trim();
    if (!line) continue;

    // Detect Sources section — only after substantial body content (10+ lines)
    // Handles: "Sources", "## Sources", "### Sources", "**Sources**", "Sources:"
    if (
      bodyLineCount >= 10 &&
      (
        /^sources\s*$/i.test(line) ||
        /^#{1,3}\s*sources\s*$/i.test(line) ||
        /^\*\*sources\*\*\s*$/i.test(line) ||
        /^sources\s*:\s*$/i.test(line)
      )
    ) {
      inSources = true;
      continue;
    }

    if (inSources) {
      // Parse: [1] Source name — URL  or  [1] Source name
      const citMatch = line.match(/^\[(\d+)\]\s+(.+)/);
      if (citMatch) {
        const num  = citMatch[1];
        const rest = citMatch[2].trim();
        // Match trailing " — https://..." or " - https://..."
        const urlMatch = rest.match(/^(.+?)\s+[—\-]+\s+(https?:\/\/\S+)$/);
        if (urlMatch) {
          const name = urlMatch[1].trim();
          const url  = urlMatch[2].trim();
          sourcesHtml += `<li><span class="src-num">[${num}]</span> <a href="${url}" target="_blank" rel="noopener">${name}</a></li>`;
        } else {
          const clean = rest.replace(/\s+[—\-]+\s*$/, '').trim();
          if (clean.startsWith('http')) {
            sourcesHtml += `<li><span class="src-num">[${num}]</span> <a href="${clean}" target="_blank" rel="noopener">${clean}</a></li>`;
          } else {
            const searchUrl = `https://www.google.com/search?q=${encodeURIComponent(clean)}`;
            sourcesHtml += `<li><span class="src-num">[${num}]</span> <a href="${searchUrl}" target="_blank" rel="noopener">${clean}</a></li>`;
          }
        }
      } else if (line.startsWith('http')) {
        // Bare URL line — make it clickable
        sourcesHtml += `<li><a href="${line}" target="_blank" rel="noopener">${line}</a></li>`;
      } else if (line.match(/^[—\-]\s+/)) {
        // LLM used "— text" bullet format inside sources
        const text = line.replace(/^[—\-]\s+/, '').trim();
        const searchUrl = `https://www.google.com/search?q=${encodeURIComponent(text)}`;
        sourcesHtml += `<li><a href="${searchUrl}" target="_blank" rel="noopener">${text}</a></li>`;
      }
      // Silently skip lines that don't match any pattern (stray punctuation, blank-looking lines)
      continue;
    }

    // --- Explicit markdown heading detection ---
    const h3Match = line.match(/^###\s+(.+)/);
    const h2Match = !h3Match && line.match(/^##\s+(.+)/);
    const h1Match = !h3Match && !h2Match && line.match(/^#\s+(.+)/);

    if (h3Match) {
      chapterCount++;
      bodyLineCount++;
      const chapterNum = String(chapterCount).padStart(2, '0');
      bodyHtml += `
        <div class="blog-chapter">
          <span class="blog-chapter-label">Chapter ${chapterNum}</span>
          <h3 class="blog-subsection-heading">${renderInline(h3Match[1])}</h3>
        </div>`;
    } else if (h2Match) {
      h2Count++;
      bodyLineCount++;
      bodyHtml += `<h2 class="blog-section-heading">${renderInline(h2Match[1])}</h2>`;
      // Images disabled for now — will re-enable with a proper image API in future
      // if (imageCount < maxImages && h2Count % 2 === 0) {
      //   const imgUrl = `https://source.unsplash.com/800x400/?${encodeURIComponent(imageKeywords)}&sig=${imageCount}`;
      //   bodyHtml += `<img src="${imgUrl}" alt="${topic}" class="blog-image" loading="lazy" />`;
      //   imageCount++;
      // }
    } else if (h1Match) {
      h2Count++;
      bodyLineCount++;
      bodyHtml += `<h2 class="blog-section-heading">${renderInline(h1Match[1])}</h2>`;
    } else if (line.startsWith('---') || line.startsWith('***')) {
      continue;
    } else {
      bodyLineCount++;
      bodyHtml += `<p>${renderInline(line)}</p>`;
    }
  }

  // Build sources block if any citations were found
  const sourcesBlock = sourcesHtml ? `
    <div class="blog-sources">
      <div class="blog-sources-title">Sources</div>
      <ol class="blog-sources-list">${sourcesHtml}</ol>
    </div>` : '';

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
    ${sourcesBlock}
    ${pullQuoteHtml}
    ${takeawayHtml}
    ${seoHtml}
  `;

  // Ensure all links in sources open in a new tab when clicked
  document.querySelectorAll('.blog-sources-list a').forEach(link => {
    link.addEventListener('click', e => {
      e.preventDefault();
      window.open(link.getAttribute('href'), '_blank', 'noopener');
    });
  });
}

function downloadPDF() {
  const topic = _lastTopic || document.getElementById('topicInput').value.trim() || 'blog';
  const filename = topic.replace(/[^a-z0-9]/gi, '_').toLowerCase().slice(0, 50) + '.pdf';
  const element = document.getElementById('blogContent');

  // Inject dark-theme styles + hide images (broken Unsplash URLs)
  const styleTag = document.createElement('style');
  styleTag.id = 'pdf-dark-styles';
  styleTag.textContent = `
    .blog-article, #blogContent { background: #111111 !important; color: #f0ece4 !important; }
    .blog-title { color: #f0ece4 !important; }
    .blog-section-heading { color: #f0ece4 !important; }
    .blog-chapter-label { color: #c9a96e !important; }
    .blog-subsection-heading { color: #f0ece4 !important; }
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
    .blog-sources { border-top: 1px solid #333 !important; }
    .blog-sources-title { color: #555550 !important; }
    .blog-sources-list li { color: #555550 !important; }
    .blog-image { display: none !important; }
  `;
  document.head.appendChild(styleTag);

  const PDF_W_MM   = 210;
  const PDF_H_MM   = 297;
  const MARGIN_MM  = 14;
  const SCALE      = 2;
  const MM_TO_PX   = 3.7795275591;

  // Reserve 6mm at the bottom for page numbers
  const PAGE_NUM_H_MM = 6;
  const contentH_mm   = PDF_H_MM - MARGIN_MM * 2 - PAGE_NUM_H_MM;
  const contentH_px   = contentH_mm * MM_TO_PX * SCALE;
  const imgW_mm       = PDF_W_MM - MARGIN_MM * 2;

  // Collect the top-offsets (relative to blogContent) of every element
  // we must NOT split across: chapter blocks, pull quotes, key takeaway, tldr
  const avoidBreakEls = element.querySelectorAll(
    '.blog-chapter, .blog-pull-quote, .blog-key-takeaway, .blog-tldr, .seo-section, .blog-sources'
  );
  const elementRect = element.getBoundingClientRect();

  // Build a sorted list of "avoid-break" top positions in canvas px
  const avoidTops = Array.from(avoidBreakEls).map(el => {
    const r = el.getBoundingClientRect();
    return (r.top - elementRect.top + window.scrollY) * SCALE;
  }).sort((a, b) => a - b);

  // Build paragraph break positions — bottom edge of every <p> in the blog body
  // Used to avoid cutting mid-paragraph
  const paraBreaks = Array.from(element.querySelectorAll('.blog-body p')).map(p => {
    const r = p.getBoundingClientRect();
    return (r.bottom - elementRect.top + window.scrollY) * SCALE;
  }).sort((a, b) => a - b);

  // Collect source links BEFORE html2canvas (positions are stable at this point)
  const sourceLinks = [];
  document.querySelectorAll('.blog-sources-list a[href]').forEach(a => {
    const r = a.getBoundingClientRect();
    sourceLinks.push({
      url:  a.getAttribute('href'),
      top:  (r.top  - elementRect.top  + window.scrollY) * SCALE,
      left: (r.left - elementRect.left) * SCALE,
      w:    r.width  * SCALE,
      h:    r.height * SCALE,
    });
  });

  html2canvas(element, {
    scale: SCALE,
    backgroundColor: '#111111',
    useCORS: true,
    logging: false,
    scrollX: 0,
    scrollY: -window.scrollY,
    windowWidth: document.documentElement.scrollWidth
  }).then(canvas => {
    const imgW_px    = canvas.width;
    const totalH_px  = canvas.height;
    const pxToMm     = imgW_mm / imgW_px;

    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF({ unit: 'mm', format: 'a4', orientation: 'portrait' });

    // Content-aware page splitting:
    // For each proposed slice boundary, check if it would cut through
    // an avoid-break element. If yes, move the boundary up to just
    // before that element's top so it starts fresh on the next page.
    function smartSplitY(currentY) {
      const proposed = currentY + contentH_px;
      if (proposed >= totalH_px) return totalH_px; // last page

      // Look for any avoid-break element that starts within 120px of
      // the proposed cut line — nudge the cut up to just before it
      const LOOKAHEAD = 120 * SCALE;
      for (const top of avoidTops) {
        if (top > proposed - LOOKAHEAD && top < proposed + LOOKAHEAD) {
          return Math.max(currentY + 1, top - 4);
        }
      }

      // Also avoid cutting mid-paragraph — scan back up to 60px to find
      // the nearest paragraph boundary (approximated as a gap in content)
      const PARA_SCAN = 60 * SCALE;
      for (const top of paraBreaks) {
        if (top <= proposed && top >= proposed - PARA_SCAN) {
          return Math.max(currentY + 1, top);
        }
      }

      return proposed;
    }

    let yOffset  = 0;
    let pageNum  = 0;

    while (yOffset < totalH_px) {
      const nextY   = smartSplitY(yOffset);
      const sliceH  = nextY - yOffset;

      const pageCanvas     = document.createElement('canvas');
      pageCanvas.width     = imgW_px;
      pageCanvas.height    = sliceH;
      pageCanvas.getContext('2d').drawImage(
        canvas, 0, yOffset, imgW_px, sliceH, 0, 0, imgW_px, sliceH
      );

      const imgData   = pageCanvas.toDataURL('image/jpeg', 0.97);
      const sliceH_mm = sliceH * pxToMm;

      if (pageNum > 0) pdf.addPage();

      pdf.addImage(imgData, 'JPEG', MARGIN_MM, MARGIN_MM, imgW_mm, sliceH_mm);

      // Page number — bottom center
      pageNum++;
      pdf.setFontSize(9);
      pdf.setTextColor(120, 110, 100);
      pdf.text(
        String(pageNum),
        PDF_W_MM / 2,
        PDF_H_MM - MARGIN_MM + 2,
        { align: 'center' }
      );

      yOffset = nextY;
    }

    // Overlay invisible clickable link annotations on top of source links.
    // We know each link's canvas-pixel position; convert to the correct
    // page + mm offset and call pdf.link() which adds a hotspot without
    // touching the image underneath.
    if (sourceLinks.length > 0) {
      // Rebuild page boundaries so we know which page each link lands on
      const pageBoundaries = [];
      let scanY = 0;
      while (scanY < totalH_px) {
        const endY = smartSplitY(scanY);
        pageBoundaries.push({ start: scanY, end: endY });
        scanY = endY;
      }

      sourceLinks.forEach(link => {
        // Find which page this link's top falls on
        for (let p = 0; p < pageBoundaries.length; p++) {
          const { start, end } = pageBoundaries[p];
          if (link.top >= start && link.top < end) {
            // Convert canvas-px coords to mm on this page
            const xMm = link.left  * pxToMm + MARGIN_MM;
            const yMm = (link.top - start) * pxToMm + MARGIN_MM;
            const wMm = link.w * pxToMm;
            const hMm = link.h * pxToMm;

            pdf.setPage(p + 1);
            pdf.link(xMm, yMm, wMm, hMm, { url: link.url });
            break;
          }
        }
      });
    }

    pdf.save(filename);
    const tag = document.getElementById('pdf-dark-styles');
    if (tag) tag.remove();
  });
}
