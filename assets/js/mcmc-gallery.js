document.addEventListener("DOMContentLoaded", function () {

  const variances = ['0.001', '0.05', '2', '8'];

  const diagnostics = [
    {
      key: 'hist',
      label: 'MCMC samples vs target',
      caption: 'MCMC samples vs normalised target g(x)/Z'
    },
    {
      key: 'trace',
      label: 'Trace plot',
      caption: 'Trace Plot'
    },
    {
      key: 'running_mean',
      label: 'Running mean',
      caption: 'Cumulative vs true mean'
    },
    {
      key: 'acf',
      label: 'ACF',
      caption: 'Autocorrelation at lags 1–100, post burn-in'
    }
  ];

  const slides = [];

  variances.forEach(v => {
    diagnostics.forEach(d => {
      slides.push({
        title: `${d.label} — var = ${v}`,
        src: `/assets/images/approximate_inference/${d.key}_var_${v}.png`,
        caption: d.caption,
        group: d.label
      });
    });
  });

  let current = 0;

  const groups = [...new Set(slides.map(s => s.group))];

  const tabsEl = document.getElementById('tabs');
  const titleEl = document.getElementById('slide-title');
  const counterEl = document.getElementById('slide-counter');
  const imgEl = document.getElementById('slide-img');
  const captionEl = document.getElementById('slide-caption');
  const btnPrev = document.getElementById('btn-prev');
  const btnNext = document.getElementById('btn-next');

  groups.forEach(g => {
    const t = document.createElement('div');

    t.className = 'group-tab';
    t.textContent = g;

    t.addEventListener('click', () => {
      goTo(slides.findIndex(s => s.group === g));
    });

    tabsEl.appendChild(t);
  });

  function render() {
    const s = slides[current];

    titleEl.textContent = s.title;
    counterEl.textContent = `${current + 1} / ${slides.length}`;

    imgEl.src = s.src;
    imgEl.alt = s.title;

    captionEl.textContent = s.caption;

    btnPrev.disabled = current === 0;
    btnNext.disabled = current === slides.length - 1;

    document.querySelectorAll('.group-tab').forEach((t, i) => {
      t.classList.toggle('active', groups[i] === slides[current].group);
    });
  }

  function goTo(i) {
    current = Math.max(0, Math.min(slides.length - 1, i));
    render();
  }

  function move(d) {
    goTo(current + d);
  }

  btnPrev.addEventListener('click', () => move(-1));
  btnNext.addEventListener('click', () => move(1));

  document.addEventListener('keydown', e => {
    if (e.key === 'ArrowRight') move(1);
    if (e.key === 'ArrowLeft') move(-1);
  });

  render();
});