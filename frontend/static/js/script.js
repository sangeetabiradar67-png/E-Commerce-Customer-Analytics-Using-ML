document.addEventListener("DOMContentLoaded", function () {
    const customersElement = document.getElementById("customers");
    const productsElement = document.getElementById("products");
    const statusElement = document.getElementById("status");
    const loadButton = document.getElementById("loadStats");
    const chartElement = document.getElementById("purchaseResults");

    function setStatus(message) {
        if (statusElement) statusElement.textContent = message;
    }

    // 1. Dashboard statistics
    async function loadAnalytics() {
        setStatus("Loading analytics...");

        try {
            const response = await fetch("/stats");
            if (!response.ok) {
                throw new Error("Failed to load dashboard statistics.");
            }

            const data = await response.json();

            if (customersElement) {
                customersElement.textContent = data.total_customers;
            }
            if (productsElement) {
                productsElement.textContent = data.total_products;
            }

            setStatus("Analytics loaded successfully!");
        } catch (error) {
            console.error("Analytics error:", error);
            setStatus("Unable to load analytics. Please check the backend.");
        }
    }

    // 2. Customer purchase analysis
    async function loadCustomerChart() {
        if (!chartElement) return;

        try {
            const response = await fetch("/customer-analysis");
            if (!response.ok) {
                throw new Error("Failed to load customer analysis.");
            }

            const data = await response.json();
            const customers = data.customers || [];
            const orders = data.total_orders || [];

            if (customers.length === 0 || orders.length === 0) {
                chartElement.textContent =
                    "No customer purchase data is available.";
                return;
            }

            const count = Math.min(10, customers.length, orders.length);

            chartElement.innerHTML = "";

            const heading = document.createElement("h3");
            heading.textContent = "Customer Purchase Analysis";
            chartElement.appendChild(heading);

            for (let i = 0; i < count; i++) {
                const row = document.createElement("p");
                row.textContent =
                    "Customer " + customers[i] + " — Orders: " + orders[i];
                chartElement.appendChild(row);
            }
        } catch (error) {
            console.error("Customer chart error:", error);
            chartElement.textContent =
                "Unable to display customer purchase analysis.";
        }
    }

    // 3. Recommendation evaluation chart
    function updateBar(id, value) {
        const bar = document.getElementById(id);
        if (!bar || !Number.isFinite(value)) return;

        bar.style.width = value + "%";
        bar.textContent = value.toFixed(2) + "%";
    }

    async function loadEvaluation() {
        try {
            const response = await fetch("/evaluation");
            if (!response.ok) {
                throw new Error("Failed to load evaluation.");
            }

            const data = await response.json();
            const model = data.item_based_cosine || {};
            const base = data.popularity_baseline || {};

            updateBar("chart-model-5", model.hit_rate_at_5 * 100);
            updateBar("chart-base-5", base.hit_rate_at_5 * 100);
            updateBar("chart-model-10", model.hit_rate_at_10 * 100);
            updateBar("chart-base-10", base.hit_rate_at_10 * 100);
        } catch (error) {
            console.error("Evaluation error:", error);
        }
    }

    // 4. Load everything
    async function loadAll() {
        await loadAnalytics();
        await loadCustomerChart();
        await loadEvaluation();
    }

    if (loadButton) {
        loadButton.addEventListener("click", loadAll);
    }

    loadAll();
});