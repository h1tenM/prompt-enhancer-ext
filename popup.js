document.addEventListener("DOMContentLoaded", () => {
	const enhanceBtn = document.getElementById("enhanceBtn");
	const copyBtn = document.getElementById("copyBtn");
	const userInput = document.getElementById("userInput");
	const outputArea = document.getElementById("outputArea");

	const loading = document.getElementById("loading");

	const metricsDashboard = document.getElementById("metricsDashboard");
	const valTokens = document.getElementById("valTokens");
	const valPrompts = document.getElementById("valPrompts");
	const valTime = document.getElementById("valTime");

	let lifetimeStats = JSON.parse(
		localStorage.getItem("promptEnhancerStats"),
	) || {
		tokens: 0,
		prompts: 0,
		time: 0,
	};

	enhanceBtn.addEventListener("click", async () => {
		const text = userInput.value.trim();
		if (!text) return alert("Please enter a prompt first!");

		enhanceBtn.disabled = true;
		loading.classList.remove("hidden");
		outputArea.classList.add("hidden");
		copyBtn.classList.add("hidden");
		metricsDashboard.classList.add("hidden");

		try {
			const persona = document.getElementById("personaSelect").value;
			const reasoning = document.getElementById("reasoningSelect").value;
			const format = document.getElementById("formatSelect").value;

			const response = await fetch("http://localhost:8000/enhance", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					prompt: text,
					persona: persona,
					reasoning: reasoning,
					format: format,
				}),
			});

			if (!response.ok) throw new Error(`Server error: ${response.status}`);

			const data = await response.json();
			// console.log("Full Backend Data:", data);

			// SAFETY CHECK: Ensure data and the property exist
			if (!data || data.enhanced_prompt === undefined) {
				throw new Error("Backend returned empty or invalid data");
			}

			// HANDLE BOTH ARRAY AND STRING FORMATS FROM API RESPONSE
			let finalResult = "";
			if (Array.isArray(data.enhanced_prompt)) {
				finalResult = data.enhanced_prompt[0].text;
			} else {
				finalResult = data.enhanced_prompt;
			}

			outputArea.value = finalResult;
			outputArea.classList.remove("hidden");
			copyBtn.classList.remove("hidden");

			// --- METRICS LOGIC ---
			if (data.metrics) {
				// Update lifetime stats
				lifetimeStats.tokens += data.metrics.tokens_saved;
				lifetimeStats.prompts += data.metrics.prompts_saved;
				lifetimeStats.time += data.metrics.time_saved_minutes;

				// Save back to local storage
				localStorage.setItem("promptEnhancerStats", JSON.stringify(lifetimeStats));

				// Display on UI (Formatted with commas for high numbers)
				valTokens.innerText = lifetimeStats.tokens.toLocaleString();
				valPrompts.innerText = lifetimeStats.prompts;
				valTime.innerText = lifetimeStats.time;

				// Reveal the dashboard
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

	copyBtn.addEventListener("click", () => {
		outputArea.select();
		document.execCommand("copy");
		copyBtn.innerText = "✅ Copied!";
		setTimeout(() => (copyBtn.innerText = "📋 Copy to Clipboard"), 2000);
	});
});
