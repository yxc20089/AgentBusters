"""
Tool definitions for Purple Agent function calling.

Defines OpenAI-compatible function schemas for all available financial analysis tools.
"""

# Core financial data tools
TOOLS = [
    # Web Search Tools
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for financial information, news, earnings data, or any other information not available in SEC filings. Use this for recent news, earnings guidance, ARPU metrics, and other data that changes frequently.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (1-10)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_financial_news",
            "description": "Search for recent financial news about a specific company. Use for earnings announcements, guidance updates, management changes, or market-moving news.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {
                        "type": "string",
                        "description": "Company name or ticker symbol"
                    },
                    "topic": {
                        "type": "string",
                        "description": "Specific topic (e.g., 'earnings', 'guidance', 'merger')",
                        "default": ""
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 5
                    }
                },
                "required": ["company"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_earnings_info",
            "description": "Search for earnings call information, guidance, and quarterly results. Use for questions about revenue guidance, EPS guidance, earnings beats/misses, and forward-looking statements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    },
                    "quarter": {
                        "type": "string",
                        "description": "Quarter (e.g., 'Q1', 'Q2', 'Q3', 'Q4')",
                        "default": ""
                    },
                    "year": {
                        "type": "integer",
                        "description": "Fiscal year",
                        "default": None
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    # Stock Data Tools
    {
        "type": "function",
        "function": {
            "name": "get_quote",
            "description": "Get current stock quote including price, market cap, P/E ratio, and other basic metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL, MSFT)"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_historical_prices",
            "description": "Get historical price data for a stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    },
                    "period": {
                        "type": "string",
                        "description": "Time period (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')",
                        "default": "1mo"
                    },
                    "interval": {
                        "type": "string",
                        "description": "Data interval (e.g., '1d', '1wk', '1mo')",
                        "default": "1d"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_financials",
            "description": "Get financial statements (income statement, balance sheet, cash flow) for a company.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    },
                    "statement_type": {
                        "type": "string",
                        "description": "Type of statement: 'income', 'balance', 'cashflow'",
                        "default": "income"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_key_statistics",
            "description": "Get key statistics including beta, profit margins, revenue growth, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_analyst_estimates",
            "description": "Get analyst price targets and recommendations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_earnings",
            "description": "Get historical earnings data and upcoming earnings estimates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    # SEC EDGAR Tools
    {
        "type": "function",
        "function": {
            "name": "get_company_info",
            "description": "Get company information from SEC EDGAR including CIK, SIC code, and basic details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_filing",
            "description": "Get SEC filing content (10-K, 10-Q, 8-K, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    },
                    "form_type": {
                        "type": "string",
                        "description": "Filing type (e.g., '10-K', '10-Q', '8-K')",
                        "default": "10-K"
                    },
                    "fiscal_year": {
                        "type": "integer",
                        "description": "Fiscal year for the filing",
                        "default": None
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_xbrl_financials",
            "description": "Get structured XBRL financial data from SEC filings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    },
                    "statement_type": {
                        "type": "string",
                        "description": "Statement type: 'income', 'balance', 'cashflow'",
                        "default": "income"
                    },
                    "fiscal_year": {
                        "type": "integer",
                        "description": "Fiscal year",
                        "default": None
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    # Calculation Tools
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code for complex calculations. Use for CAGR, ratios, or any custom financial calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Must print the result."
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Execution timeout in seconds",
                        "default": 30
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_financial_metric",
            "description": "Calculate common financial metrics (CAGR, ROE, ROA, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "description": "Metric to calculate: 'cagr', 'roe', 'roa', 'current_ratio', 'debt_to_equity', 'profit_margin', 'eps', 'pe_ratio'"
                    },
                    "values": {
                        "type": "object",
                        "description": "Input values for the calculation (e.g., {'start_value': 100, 'end_value': 150, 'years': 3} for CAGR)"
                    }
                },
                "required": ["metric", "values"]
            }
        }
    },
    # Options Tools
    {
        "type": "function",
        "function": {
            "name": "get_options_chain",
            "description": "Get options chain data for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    },
                    "expiration": {
                        "type": "string",
                        "description": "Expiration date (YYYY-MM-DD) or 'nearest'",
                        "default": "nearest"
                    },
                    "min_strike": {
                        "type": "number",
                        "description": "Minimum strike price",
                        "default": None
                    },
                    "max_strike": {
                        "type": "number",
                        "description": "Maximum strike price",
                        "default": None
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_option_price",
            "description": "Calculate theoretical option price using Black-Scholes model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "spot_price": {
                        "type": "number",
                        "description": "Current stock price"
                    },
                    "strike_price": {
                        "type": "number",
                        "description": "Option strike price"
                    },
                    "days_to_expiry": {
                        "type": "integer",
                        "description": "Days until expiration"
                    },
                    "volatility": {
                        "type": "number",
                        "description": "Annualized volatility (e.g., 0.25 for 25%)"
                    },
                    "option_type": {
                        "type": "string",
                        "description": "'call' or 'put'",
                        "default": "call"
                    },
                    "risk_free_rate": {
                        "type": "number",
                        "description": "Risk-free interest rate",
                        "default": 0.05
                    }
                },
                "required": ["spot_price", "strike_price", "days_to_expiry", "volatility"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_volatility_analysis",
            "description": "Get volatility analysis including historical and implied volatility.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    },
                    "lookback_days": {
                        "type": "integer",
                        "description": "Number of days for historical volatility calculation",
                        "default": 30
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_options_strategy",
            "description": "Analyze an options strategy (straddle, spread, iron condor, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "legs": {
                        "type": "array",
                        "description": "List of option legs, each with type, strike, expiry, quantity",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "description": "'call' or 'put'"},
                                "strike": {"type": "number"},
                                "expiry": {"type": "string"},
                                "quantity": {"type": "integer", "description": "Positive for long, negative for short"}
                            }
                        }
                    },
                    "spot_price": {
                        "type": "number",
                        "description": "Current stock price"
                    }
                },
                "required": ["legs", "spot_price"]
            }
        }
    },
    # Risk Tools
    {
        "type": "function",
        "function": {
            "name": "calculate_portfolio_greeks",
            "description": "Calculate aggregate Greeks for a portfolio of options positions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "positions": {
                        "type": "array",
                        "description": "List of positions with delta, gamma, theta, vega, quantity",
                        "items": {
                            "type": "object"
                        }
                    }
                },
                "required": ["positions"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_var",
            "description": "Calculate Value at Risk for a portfolio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "returns": {
                        "type": "array",
                        "description": "Array of historical returns",
                        "items": {"type": "number"}
                    },
                    "confidence_level": {
                        "type": "number",
                        "description": "Confidence level (e.g., 0.95 for 95%)",
                        "default": 0.95
                    },
                    "portfolio_value": {
                        "type": "number",
                        "description": "Current portfolio value",
                        "default": 100000
                    }
                },
                "required": ["returns"]
            }
        }
    },
    # FAB Benchmark Data Tool
    {
        "type": "function",
        "function": {
            "name": "search_fab_benchmark",
            "description": "Search the FAB (Finance Agent Benchmark) dataset for specific financial data. This contains curated financial data including Netflix ARPU, AMD guidance, company metrics, and other benchmark data. Use this for questions about specific financial metrics, earnings guidance, ARPU trends, and similar benchmark data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant benchmark data (e.g., 'Netflix ARPU', 'AMD revenue guidance', 'TJX margin')"
                    },
                    "company": {
                        "type": "string",
                        "description": "Optional company name or ticker to filter results",
                        "default": ""
                    }
                },
                "required": ["query"]
            }
        }
    },
    # Reference File Tools
    {
        "type": "function",
        "function": {
            "name": "fetch_reference_file",
            "description": "Fetch and parse a reference file from a URL. Supports PDF, Excel (.xlsx, .xls), Word (.docx), CSV, JSON, images, and plain text files. Use this to access reference materials provided for the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the file to fetch"
                    },
                    "format_hint": {
                        "type": "string",
                        "description": "Optional format hint if URL doesn't have extension (pdf, xlsx, docx, csv, json, txt, image)",
                        "default": None
                    },
                    "extract_tables": {
                        "type": "boolean",
                        "description": "For PDFs/documents, whether to extract tables as structured data",
                        "default": True
                    },
                    "page_start": {
                        "type": "integer",
                        "description": "For PDFs, starting page number (1-indexed)",
                        "default": 1
                    },
                    "page_limit": {
                        "type": "integer",
                        "description": "For PDFs, maximum number of pages to extract (None for all)",
                        "default": None
                    },
                    "row_offset": {
                        "type": "integer",
                        "description": "For Excel/CSV, number of rows to skip",
                        "default": 0
                    },
                    "row_limit": {
                        "type": "integer",
                        "description": "For Excel/CSV, maximum number of rows to return (None for all)",
                        "default": None
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_reference_files",
            "description": "List all available reference files for this task with their URLs and metadata (file type, size, description).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
]

# Tool name to function mapping (will be set by executor)
TOOL_HANDLERS = {}
