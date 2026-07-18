const API = "http://localhost:8000";

document.addEventListener("DOMContentLoaded", () => {
	const enhanceBtn = document.getElementById("enhanceBtn");
	const copyBtn = document.getElementById("copyBtn");
	const userInput = document.getElementById("userInput");
	const outputArea = document.getElementById("outputArea");

	const loading = document.getElementById("loading");

	const metricsDashboard = document.getElementById("metricsDashboard");
	const metricsNote = document.getElementById("metricsNote");
	const valWinRate = document.getElementById("valWinRate");
	const valTokens = document.getElementById("valTokens");
	const valCount = document.getElementById("valCount");

	const sourcesToggle = document.getElementById("sourcesToggle");
	const sourcesPanel = document.getElementById("sourcesPanel");
	const sourcesList = document.getElementById("sourcesList");
	const sourcesCount = document.getElementById("sourcesCount");
	const sourcesStatus = document.getElementById("sourcesStatus");
	const newDocName = document.getElementById("newDocName");
	const newDocText = document.getElementById("newDocText");
	const newDocFile = document.getElementById("newDocFile");
	const fileLabelText = document.getElementById("fileLabelText");
	const addDocBtn = document.getElementById("addDocBtn");

	let lifetimeStats = { tokens: 0, count: 0 };

	chrome.storage.local.get("promptEnhancerStats", (result) => {
		if (result.promptEnhancerStats) {
			lifetimeStats = result.promptEnhancerStats;
		}
	});

	const saveStats = () =>
		chrome.storage.local.set({ promptEnhancerStats: lifetimeStats });

	// --- knowledge base ------------------------------------------------

	function showStatus(message, isError = false) {
		sourcesStatus.innerText = message;
		sourcesStatus.classList.remove("hidden");
		sourcesStatus.classList.toggle("error", isError);
		setTimeout(() => sourcesStatus.classList.add("hidden"), 3000);
	}

	function renderDocs(docs) {
		const active = docs.filter((d) => d.enabled).length;
		sourcesCount.innerText = `(${active}/${docs.length} active)`;

		if (!docs.length) {
			sourcesList.innerHTML =
				'<div class="sources-empty">No documents yet. Add one below.</div>';
			return;
		}

		sourcesList.innerHTML = "";
		for (const doc of docs) {
			const row = document.createElement("div");
			row.className = "source-row";

			const toggle = document.createElement("input");
			toggle.type = "checkbox";
			toggle.checked = doc.enabled;
			toggle.addEventListener("change", () =>
				setEnabled(doc.id, toggle.checked),
			);

			const label = document.createElement("span");
			label.className = "source-name";
			label.innerText = doc.name;
			label.title = `${doc.chunks} chunks · ${doc.chars.toLocaleString()} chars`;

			const remove = document.createElement("button");
			remove.className = "remove-btn";
			remove.innerText = "×";
			remove.title = "Remove document";
			remove.addEventListener("click", () => deleteDoc(doc.id, doc.name));

			row.append(toggle, label, remove);
			sourcesList.appendChild(row);
		}
	}

	async function loadDocs() {
		try {
			const res = await fetch(`${API}/sources`);
			if (!res.ok) throw new Error(`Server error: ${res.status}`);
			const data = await res.json();
			renderDocs(data.docs);
		} catch (error) {
			sourcesList.innerHTML = `<div class="sources-empty error">⚠️ ${error.message}<br/>Is the backend running?</div>`;
			sourcesCount.innerText = "";
		}
	}

	async function setEnabled(id, enabled) {
		try {
			const res = await fetch(`${API}/sources/${id}`, {
				method: "PATCH",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ enabled }),
			});
			if (!res.ok) throw new Error(`Server error: ${res.status}`);
			await loadDocs();
		} catch (error) {
			showStatus(error.message, true);
			await loadDocs();
		}
	}

	async function deleteDoc(id, name) {
		if (!confirm(`Remove "${name}" from the knowledge base?`)) return;
		try {
			const res = await fetch(`${API}/sources/${id}`, { method: "DELETE" });
			if (!res.ok) throw new Error(`Server error: ${res.status}`);
			showStatus(`Removed "${name}".`);
			await loadDocs();
		} catch (error) {
			showStatus(error.message, true);
		}
	}

	newDocFile.addEventListener("change", () => {
		const file = newDocFile.files[0];
		fileLabelText.innerText = file ? file.name : "Choose .txt file";
		if (file && !newDocName.value.trim()) {
			newDocName.value = file.name.replace(/\.txt$/i, "");
		}
	});

	addDocBtn.addEventListener("click", async () => {
		const name = newDocName.value.trim();
		const text = newDocText.value.trim();
		const file = newDocFile.files[0];

		if (!name) return showStatus("Give the document a name.", true);
		if (!text && !file)
			return showStatus("Paste some text or choose a file.", true);

		const form = new FormData();
		form.append("name", name);
		if (file) form.append("file", file);
		else form.append("text", text);

		addDocBtn.disabled = true;
		addDocBtn.innerText = "Embedding…";

		try {
			const res = await fetch(`${API}/sources`, { method: "POST", body: form });
			if (!res.ok) {
				const detail = await res.json().catch(() => null);
				throw new Error(detail?.detail || `Server error: ${res.status}`);
			}
			const doc = await res.json();
			showStatus(`Added "${doc.name}" (${doc.chunk_ids.length} chunks).`);

			newDocName.value = "";
			newDocText.value = "";
			newDocFile.value = "";
			fileLabelText.innerText = "Choose .txt file";
			await loadDocs();
		} catch (error) {
			showStatus(error.message, true);
		} finally {
			addDocBtn.disabled = false;
			addDocBtn.innerText = "+ Add Document";
		}
	});

	sourcesToggle.addEventListener("click", () => {
		sourcesPanel.classList.toggle("hidden");
		if (!sourcesPanel.classList.contains("hidden")) loadDocs();
	});

	loadDocs();

	// --- enhancement ---------------------------------------------------

	enhanceBtn.addEventListener("click", async () => {
		const text = userInput.value.trim();
		if (!text) return alert("Please enter a prompt first!");

		enhanceBtn.disabled = true;
		loading.classList.remove("hidden");
		outputArea.classList.add("hidden");
		copyBtn.classList.add("hidden");
		metricsDashboard.classList.add("hidden");
		metricsNote.classList.add("hidden");

		try {
			const persona = document.getElementById("personaSelect").value;
			const reasoning = document.getElementById("reasoningSelect").value;
			const format = document.getElementById("formatSelect").value;

			const response = await fetch(`${API}/enhance`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ prompt: text, persona, reasoning, format }),
			});

			if (!response.ok) throw new Error(`Server error: ${response.status}`);

			const data = await response.json();
			if (!data || data.enhanced_prompt === undefined) {
				throw new Error("Backend returned empty or invalid data");
			}

			outputArea.value = data.enhanced_prompt;
			outputArea.classList.remove("hidden");
			copyBtn.classList.remove("hidden");

			// --- METRICS LOGIC ---
			if (data.metrics) {
				lifetimeStats.tokens += data.metrics.enhancement_tokens || 0;
				lifetimeStats.count += 1;
				saveStats();

				valTokens.innerText = lifetimeStats.tokens.toLocaleString();
				valCount.innerText = lifetimeStats.count;

				// Win rate is measured offline by the eval harness, not guessed here.
				const evalData = data.metrics.eval;
				if (evalData?.win_rate_pct != null) {
					valWinRate.innerText = `${evalData.win_rate_pct}%`;
					metricsNote.innerText = `Win rate measured offline on ${evalData.n_prompts} prompts vs. un-enhanced baseline (${evalData.judge_model} judge, position-swapped).`;
					metricsNote.classList.remove("hidden");
				} else {
					valWinRate.innerText = "—";
					metricsNote.innerText =
						"Run backend/eval to populate the measured win rate.";
					metricsNote.classList.remove("hidden");
				}

				metricsDashboard.classList.remove("hidden");
			}
		} catch (error) {
			console.error("Detailed Error:", error);
			outputArea.value = "⚠️ " + error.message;
			outputArea.classList.remove("hidden");
		} finally {
			enhanceBtn.disabled = false;
			loading.classList.add("hidden");
		}
	});

	copyBtn.addEventListener("click", async () => {
		try {
			await navigator.clipboard.writeText(outputArea.value);
			copyBtn.innerText = "✅ Copied!";
		} catch {
			copyBtn.innerText = "⚠️ Copy failed";
		}
		setTimeout(() => (copyBtn.innerText = "📋 Copy to Clipboard"), 2000);
	});
});
