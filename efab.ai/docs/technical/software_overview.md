# Harness the Power of AI for Your Supply Chain: An Overview of the Beverly Knits Raw Material Planner

**Transform your raw material planning from a complex challenge into a strategic advantage.**

The Beverly Knits AI Raw Material Planner is a state-of-the-art, intelligent system designed to revolutionize how textile manufacturers manage their supply chains. By leveraging the power of artificial intelligence, our planner provides precise, cost-effective, and timely procurement recommendations, ensuring you have the right materials, in the right quantity, at the right time.

---

## Key Features: At a Glance

| Feature | Description | Benefit |
| :--- | :--- | :--- |
| **Intelligent Forecasting** | Aggregates multiple demand forecasts with source reliability weighting. | More accurate demand prediction, reducing overstock and stockouts. |
| **Automated BOM Explosion**| Translates finished good forecasts into detailed raw material requirements. | Eliminates manual calculations and ensures no material is overlooked. |
| **Real-Time Inventory Netting**| Analyzes current stock and open purchase orders to determine net requirements. | Prevents unnecessary purchases and optimizes cash flow. |
| **Advanced Optimization** | Applies safety stock, MOQs, and EOQ to find the most cost-effective order quantities. | Minimizes holding costs and takes advantage of economic order quantities. |
| **Smart Supplier Selection**| Recommends the best suppliers based on cost, reliability, and risk diversification. | Reduces supply chain risk and ensures a reliable supply of materials. |
| **User-Friendly Dashboard** | An interactive Streamlit interface for easy operation and data visualization. | Empowers your team to make data-driven decisions with ease. |

---

## How It Works: A Six-Step Process to Perfection

Our system simplifies the complexity of raw material planning into a seamless, automated, six-step process:

1.  **Forecast Unification:** We start by intelligently combining all your sales forecasts. Our system weighs each forecast based on its historical accuracy, giving you a single, reliable demand signal.
2.  **BOM Explosion:** The system automatically breaks down the demand for finished goods into the specific raw materials required, handling complex bills of materials and unit conversions effortlessly.
3.  **Inventory Netting:** We then look at what you already have. The planner subtracts available inventory and incoming stock from the gross requirements to determine exactly what you need to order.
4.  **Procurement Optimization:** This is where the magic happens. The system applies sophisticated rules, including safety stock for demand variability, minimum order quantities (MOQs), and Economic Order Quantity (EOQ) principles to calculate the optimal amount to purchase.
5.  **Supplier Selection:** Our AI evaluates your suppliers based on cost, lead time, and reliability, recommending the best sourcing strategy to balance cost and risk. It even supports multi-supplier strategies to enhance supply chain resilience.
6.  **Output Generation:** Finally, the planner generates a clear, actionable report detailing what to order, from whom, and when. Each recommendation is backed by a clear rationale, giving you complete transparency.

---

## A Look Under the Hood: Technical Architecture

Our AI Planner is built on a robust, modern technology stack, designed for performance, scalability, and maintainability.

### Technology Stack
- **Backend:** Python 3.12
- **Core Libraries:** Pandas (for data manipulation), NumPy (for numerical operations), Pydantic (for robust data modeling), Loguru (for logging).
- **Web Interface:** Streamlit, a fast and easy way to build beautiful data apps.
- **Deployment:** Docker, Docker Compose, and Systemd for flexible deployment options from local development to production servers.

### Modular by Design
The system features a clean, modular architecture that separates concerns, making it easy to maintain and extend:
-   `engine/`: Contains the core business logic for the 6-step planning process.
-   `models/`: Defines the robust Pydantic data models that power the application.
-   `utils/`: Houses shared utility functions used across the application.
-   `scripts/`: Stores scripts for data integration, database maintenance, and other operational tasks.

### Core Data Models
The planner's logic is built around a set of well-defined data models:
1.  `FinishedGoodsForecast`: Manages multi-source demand forecasts.
2.  `BillOfMaterials`: Handles complex material requirements and unit conversions.
3.  `Inventory`: Tracks stock levels, including open purchase orders.
4.  `Supplier`: Manages supplier data, including performance and cost.
5.  `ProcurementRecommendation`: Structures the final, detailed output recommendations.

### Project Structure
```
beverly-knits/
├── config/          # Configuration files
├── data/            # Data files and samples
├── engine/          # Core planning engine
├── models/          # Data models
├── utils/           # Utility functions
├── tests/           # All test files (organized)
├── scripts/         # Utility and maintenance scripts
│   └── fixes/       # One-time fix scripts
├── main.py          # Main Streamlit application
└── README.md        # Project documentation
```

---

## Advanced Capabilities: Your Competitive Edge

### Unparalleled Data Integration: Turn Data Chaos into a Strategic Asset
Bad data costs businesses millions in errors, missed opportunities, and wasted time. Our planner tackles this head-on. The intelligent data integration pipeline is more than just a feature; it's your foundation for reliable, data-driven decision-making.

Imagine a world where you no longer have to second-guess your inventory numbers or spend hours manually cleaning spreadsheets. Our system automatically:
-   **Eliminates Errors at the Source:** Corrects negative inventory, standardizes cost data, and validates BOM percentages, preventing costly procurement mistakes before they happen.
-   **Builds a Single Source of Truth:** Integrates disparate data sources into a clean, unified view of your supply chain.
-   **Provides Total Transparency:** Generates detailed data quality reports, so you always know not just what your data says, but how reliable it is.

This isn't just about clean data; it's about unlocking the confidence to act decisively.

### AI-Supercharged with Zen-MCP: Step into the Future of Supply Chain AI
For organizations ready to lead the pack, our optional Zen-MCP server integration unlocks the next generation of artificial intelligence. This transforms your planner from a powerful analytical tool into a strategic AI powerhouse.

By connecting to an ecosystem of world-class AI models (like Claude, Gemini, and GPT-4), you can:
-   **Achieve Unprecedented Strategic Insight:** Leverage multi-model consensus to solve your most complex challenges, from demand forecasting in volatile markets to optimizing global supplier networks.
-   **Automate Complex Workflows:** Deploy AI agents to orchestrate sophisticated tasks, turning months of strategic planning into days of automated execution.
-   **Gain a True Competitive Advantage:** Harness the absolute cutting edge of AI to outmaneuver competitors and build a more agile, intelligent, and resilient supply chain.

---

## Why Choose the Beverly Knits AI Planner?

*   **Dramatically Reduce Operational Costs:** Stop tying up capital in excess inventory. Our planner optimizes every purchase, minimizing carrying costs and waste. By leveraging EOQ and smart supplier selection, you ensure you're not just buying what you need, but buying it in the most economically advantageous way.

*   **Boost Team Productivity and Efficiency:** Free your skilled team from the drudgery of manual data entry and spreadsheet management. The AI Planner automates hours of tedious work, allowing your planners and buyers to focus on high-value strategic activities like negotiating with suppliers and managing exceptions.

*   **Build a More Resilient Supply Chain:** In today's volatile market, resilience is paramount. Our planner helps you mitigate risk by identifying optimal safety stock levels, diversifying your sourcing across multiple suppliers, and providing the foresight needed to navigate disruptions with confidence.

*   **Make Decisions with Unprecedented Clarity:** Move from guesswork and intuition to data-backed certainty. The planner provides a single, clear view of your entire raw material landscape, with actionable recommendations and transparent reasoning. Empower your team to make faster, smarter decisions at every turn.

*   **Future-Proof Your Operations:** Built on a modern, scalable, and modular architecture, the Beverly Knits AI Planner is designed to grow with you. As your business evolves, the planner can be easily extended and adapted to meet new challenges, ensuring it remains a core strategic asset for years to come.

---

## Get Started Today

The Beverly Knits AI Raw Material Planner is more than just software; it's a partner in your success. It's production-ready and has already undergone extensive testing and refinement.

**Ready to take control of your supply chain? Let's talk.**
