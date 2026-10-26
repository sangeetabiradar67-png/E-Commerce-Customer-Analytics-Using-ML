
document.addEventListener("DOMContentLoaded", function () {
    const customersElement = document.getElementById("customers");
    const productsElement = document.getElementById("products");
    const statusElement = document.getElementById("status");
    const loadButton = document.getElementById("loadStats");
    const chartElement = document.getElementById("purchaseResults");

    // Load dashboard statistics
    async function loadAnalytics() {
        statusElement.textContent = "Loading analytics...";

        try {
            const response = await fetch("/stats");

            if (!response.ok) {
                throw new Error("Failed to load dashboard statistics.");
            }

            const data = await response.json();

            customersElement.textContent = data.total_customers;
            productsElement.textContent = data.total_products;

            statusElement.textContent = "Analytics loaded successfully!";

            // Load customer purchase analysis
            await loadCustomerChart();

        } catch (error) {
            console.error("Analytics error:", error);
            statusElement.textContent =
                "Unable to load analytics. Please check the backend.";
        }
    }

    // Load customer purchase analysis from Flask
    async function loadCustomerChart() {
        try {
            const response = await fetch("/customer-analysis");

            if (!response.ok) {
                throw new Error("Failed to load customer analysis.");
            }

            const data = await response.json();

            if (!chartElement) {
                console.warn(
                    'Element with id "customerChart" was not found.'
                );
                return;
            }

            // Display the first 10 customers as a simple text chart
            const customers = data.customers || [];
            const orders = data.total_orders || [];

            if (customers.length === 0 || orders.length === 0) {
                chartElement.textContent =
                    "No customer purchase data is available.";
                return;
            }

            const count = Math.min(
                10,
                customers.length,
                orders.length
            );

            chartElement.innerHTML = "";

            const heading = document.createElement("h3");
            heading.textContent = "Customer Purchase Analysis";
            chartElement.appendChild(heading);

            for (let i = 0; i < count; i++) {
                const row = document.createElement("p");

                row.textContent =
                    "Customer " + customers[i] +
                    " — Orders: " + orders[i];

                chartElement.appendChild(row);
            }

        } catch (error) {
            console.error("Customer chart error:", error);

            if (chartElement) {
                chartElement.textContent =
                    "Unable to display customer purchase analysis.";
            }
        }
    }

    // Connect the Load Analytics button
    if (loadButton) {
        loadButton.addEventListener("click", loadAnalytics);
    } else {
        console.error('Button with id "loadStats" was not found.');
    }

    // Load dashboard statistics automatically on page open
    loadAnalytics();
});
