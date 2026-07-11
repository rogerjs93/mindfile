/* portfolio motion — reduced-motion aware, null-guarded (shared by index + showcase). */
(() => {
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---- theme toggle ---- */
  const root = document.documentElement;
  const saved = localStorage.getItem('theme');
  if (saved) root.setAttribute('data-theme', saved);
  const toggle = document.querySelector('[data-theme-toggle]');
  if (toggle) {
    const sync = () => {
      const isDark = (root.getAttribute('data-theme') ||
        (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')) === 'dark';
      toggle.textContent = isDark ? '☀' : '☾';
      toggle.setAttribute('aria-label', isDark ? 'Switch to light theme' : 'Switch to dark theme');
    };
    sync();
    toggle.addEventListener('click', () => {
      const cur = root.getAttribute('data-theme') ||
        (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
      const next = cur === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      localStorage.setItem('theme', next);
      sync();
    });
  }

  /* ---- reveal on scroll (content is visible by default; this only enhances) ---- */
  const reveals = document.querySelectorAll('.reveal');
  if (reveals.length && !reduce && 'IntersectionObserver' in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
    }, { rootMargin: '0px 0px -8% 0px' });
    reveals.forEach((el) => io.observe(el));
  } else {
    reveals.forEach((el) => el.classList.add('in'));
  }

  /* ---- category filter (index) ---- */
  const filter = document.querySelector('[data-filter]');
  if (filter) {
    const tiles = [...document.querySelectorAll('[data-category]')];
    const sections = [...document.querySelectorAll('[data-cat-section]')];
    filter.addEventListener('click', (e) => {
      const btn = e.target.closest('.chip-btn');
      if (!btn) return;
      const cat = btn.dataset.cat;
      filter.querySelectorAll('.chip-btn').forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
      tiles.forEach((t) => {
        const show = cat === 'all' || t.dataset.category === cat;
        t.style.transition = reduce ? 'none' : 'opacity .3s var(--ease), transform .3s var(--ease)';
        t.style.opacity = show ? '1' : '0';
        t.style.transform = show ? 'none' : 'scale(.98)';
        t.style.display = show ? '' : 'none';
      });
      sections.forEach((s) => {
        const has = s.querySelector('[data-category]:not([style*="display: none"])');
        s.style.display = (cat === 'all' || s.dataset.catSection === cat) ? '' : 'none';
      });
    });
  }

  /* ---- carousels (featured + screenshots): dots + keyboard ---- */
  document.querySelectorAll('[data-carousel]').forEach((car) => {
    const track = car.querySelector('.track');
    const dotsWrap = car.querySelector('.dots');
    if (!track) return;
    const slides = [...track.children];
    if (dotsWrap) {
      slides.forEach((_, i) => {
        const d = document.createElement('button');
        d.type = 'button';
        d.setAttribute('aria-label', `Go to item ${i + 1}`);
        if (i === 0) d.setAttribute('aria-current', 'true');
        d.addEventListener('click', () => slides[i].scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', inline: 'center', block: 'nearest' }));
        dotsWrap.appendChild(d);
      });
      const dots = [...dotsWrap.children];
      const io = new IntersectionObserver((entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            const i = slides.indexOf(e.target);
            dots.forEach((d, j) => d.setAttribute('aria-current', String(j === i)));
          }
        });
      }, { root: track, threshold: 0.6 });
      slides.forEach((s) => io.observe(s));
    }
    car.setAttribute('tabindex', '0');
    car.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      e.preventDefault();
      track.scrollBy({ left: (e.key === 'ArrowRight' ? 1 : -1) * track.clientWidth * 0.8, behavior: reduce ? 'auto' : 'smooth' });
    });
  });

  /* ---- parallax hero canvas (drifting instrument motif) ---- */
  const canvas = document.querySelector('[data-hero-canvas]');
  if (canvas && !reduce) {
    const ctx = canvas.getContext('2d');
    let w, h, nodes, raf, scrollY = 0;
    const N = 34;
    const teal = () => getComputedStyle(root).getPropertyValue('--primary').trim() || 'rgb(90,170,180)';
    const resize = () => {
      const r = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = canvas.width = r.width * dpr; h = canvas.height = r.height * dpr;
      ctx.scale(dpr, dpr);
      nodes = Array.from({ length: N }, () => ({
        x: Math.random() * r.width, y: Math.random() * r.height,
        vx: (Math.random() - 0.5) * 0.18, vy: (Math.random() - 0.5) * 0.18, z: Math.random()
      }));
      canvas._rw = r.width; canvas._rh = r.height;
    };
    const draw = () => {
      const rw = canvas._rw, rh = canvas._rh;
      ctx.clearRect(0, 0, w, h);
      const color = teal();
      const py = scrollY * 0.12;
      for (const n of nodes) {
        n.x += n.vx; n.y += n.vy;
        if (n.x < 0 || n.x > rw) n.vx *= -1;
        if (n.y < 0 || n.y > rh) n.vy *= -1;
      }
      ctx.globalAlpha = 0.5;
      ctx.strokeStyle = color; ctx.lineWidth = 1;
      for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        const dx = a.x - b.x, dy = (a.y - b.y);
        const d = Math.hypot(dx, dy);
        if (d < 130) {
          ctx.globalAlpha = (1 - d / 130) * 0.28;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y - py * a.z); ctx.lineTo(b.x, b.y - py * b.z); ctx.stroke();
        }
      }
      ctx.fillStyle = color;
      for (const n of nodes) {
        ctx.globalAlpha = 0.35 + n.z * 0.4;
        ctx.beginPath(); ctx.arc(n.x, n.y - py * n.z, 1.6 + n.z * 1.6, 0, Math.PI * 2); ctx.fill();
      }
      raf = requestAnimationFrame(draw);
    };
    const onScroll = () => { scrollY = window.scrollY; };
    window.addEventListener('resize', () => { cancelAnimationFrame(raf); resize(); draw(); }, { passive: true });
    window.addEventListener('scroll', onScroll, { passive: true });
    resize(); draw();
  }
})();
