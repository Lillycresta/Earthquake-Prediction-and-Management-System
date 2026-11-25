document.addEventListener('DOMContentLoaded', () => {
  const toggle = document.querySelector('.nav-toggle');
  const links = document.querySelector('.nav-links');
  if (toggle && links) {
    toggle.addEventListener('click', () => {
      const open = links.classList.toggle('open');
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
  }

  const form = document.querySelector('#prediction-form');
  const result = document.querySelector('#prediction-result');
  if (form && result) {
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const submit = form.querySelector('button[type="submit"]');
      submit.disabled = true;
      submit.textContent = 'Classifying...';
      const payload = Object.fromEntries(new FormData(form));

      try {
        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Prediction failed');

        const probabilityLabels = ['Low', 'Moderate', 'High'];
        const probabilityValues = data.probabilities || {};
        const probabilityRows = probabilityLabels.map((label) => {
          const value = Number(probabilityValues[label] || 0);
          return `
            <div style="text-align:left;margin:10px 0">
              <div style="display:flex;justify-content:space-between;font-size:13px"><strong>${label}</strong><span>${value.toFixed(1)}%</span></div>
              <div class="probability-bar"><span style="width:${value}%"></span></div>
            </div>`;
        }).join('');
        const predictedCategory = data.category;
        const confidence = Number(data.confidence || probabilityValues[predictedCategory] || 0);

        result.innerHTML = `
          <span class="panel-label">RESULT</span>
          <div class="result-category">
            <p>Historical magnitude category</p>
            <strong>${predictedCategory}</strong>
            <p style="margin:4px 0 10px;color:#147d64;font-weight:700">Confidence: ${confidence.toFixed(1)}%</p>
            ${probabilityRows}
            <p style="color:#65756f;margin-top:16px">${data.advice}</p>
            <small>Model: ${data.model}</small>
          </div>`;
      } catch (error) {
        result.innerHTML = `<span class="panel-label">RESULT</span><div class="result-placeholder"><h3>Unable to classify</h3><p>${error.message}</p></div>`;
      } finally {
        submit.disabled = false;
        submit.textContent = 'Classify Magnitude';
      }
    });
  }
});
