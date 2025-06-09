const API = {
  listPrompts:  () => fetch('/prompts').then(r => r.json()),
  addPrompt:   (text) => fetch('/prompts', {
                    method: 'POST',
                    headers: {'Content-Type':'application/json'},
                    body: JSON.stringify({ prompt_text: text })
                  }).then(r => r.json()),
  getDetails:  (id) => fetch(`/prompts/${id}/worst`).then(r => r.json()),
  deletePrompt: (id) => fetch(`/prompts/${id}`, {method:'DELETE'}).then(r => r.json())
};

document.getElementById('prompt-form').onsubmit = async e => {
  e.preventDefault();
  const txt = document.getElementById('prompt-input').value.trim();
  if (!txt) return alert('Enter a prompt!');
  document.getElementById('running-indicator').style.display = 'inline-block';
  await API.addPrompt(txt);
  document.getElementById('prompt-input').value = '';
  await refreshList();
  document.getElementById('running-indicator').style.display = 'none';
};

async function refreshList() {
  const prompts = await API.listPrompts();
  const tbody = document.getElementById('prompt-rows');
  tbody.innerHTML = '';
  // assume prompts already sorted by final_score desc
    prompts.forEach((p, i) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${i + 1}</td>
        <td><a href="#" onclick="showDetails('${p.prompt_id}')">${p.prompt_id}</a></td>
        <td>${new Date(p.created_at).toLocaleString()}</td>
        <td>${(p.metrics.right_error_description * 100).toFixed(1)}%</td>
        <td>${(p.metrics.correct_hint * 100).toFixed(1)}%</td>
        <td>${(p.metrics.correct_or_absent_code * 100).toFixed(1)}%</td>
        <td>${(p.metrics.correct_line_reference * 100).toFixed(1)}%</td>
        <td>${(p.metrics.final_score * 100).toFixed(1)}%</td>
        <td><button onclick="deletePrompt('${p.prompt_id}')">Delete</button></td>
      `;
      tbody.appendChild(tr);
    });
}

async function showDetails(promptId) {
  const data = await API.getDetails(promptId);
  document.getElementById('modal-prompt-id').textContent = promptId;
  const tbody = document.getElementById('modal-rows');
  tbody.innerHTML = '';
  data.forEach(r => {
    const flags = Object.entries(r.flags)
                        .map(([k,v])=> `${k}:${v? '✔':'✖'}`)
                        .join(', ');
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${r.case_id}</td>
      <td><pre>${r.answer}</pre></td>
      <td>${flags}</td>
    `;
    tbody.appendChild(tr);
  });
  document.getElementById('details-modal').style.display = 'block';
}

function closeModal() {
  document.getElementById('details-modal').style.display = 'none';
}

async function deletePrompt(id) {
  if (!confirm('Delete this prompt?')) return;
  await API.deletePrompt(id);
  await refreshList();
}

// initial load
refreshList();
