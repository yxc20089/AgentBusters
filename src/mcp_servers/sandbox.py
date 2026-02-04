"""
Python Sandbox MCP Server

A FastMCP server that provides isolated Python code execution
with pre-loaded financial libraries for quantitative analysis.
"""

import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field


class ExecutionResult(BaseModel):
    """Result of code execution."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    execution_time_ms: int = 0
    error_type: str | None = None
    error_message: str | None = None
    variables: dict[str, Any] = Field(default_factory=dict)


# Pre-loaded libraries available in sandbox
PRELOADED_IMPORTS = """
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
"""

# Allowed modules for sandboxed import
ALLOWED_MODULES = {
    "numpy", "np",
    "pandas", "pd",
    "math",
    "datetime",
    "scipy", "scipy.stats",
    "statistics",
    "json",  # Safe for serialization
    "re",    # Regex for text processing
    "collections",  # Counter, defaultdict, etc.
    "itertools",  # Combinatorics
    "functools",  # reduce, partial
    "decimal",  # Precise decimal arithmetic
    "fractions",  # Rational numbers
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Safe import that only allows pre-approved modules."""
    # Check if this is an allowed module
    if name not in ALLOWED_MODULES and not any(name.startswith(m + ".") for m in ALLOWED_MODULES):
        raise ImportError(f"Module '{name}' is not allowed in sandbox. "
                         f"Allowed: {', '.join(sorted(ALLOWED_MODULES))}")
    return __builtins__["__import__"](name, globals, locals, fromlist, level)


# Safe builtins for execution
SAFE_BUILTINS = {
    'abs': abs,
    'all': all,
    'any': any,
    'bool': bool,
    'dict': dict,
    'enumerate': enumerate,
    'filter': filter,
    'float': float,
    'format': format,
    'frozenset': frozenset,
    'int': int,
    'isinstance': isinstance,
    'len': len,
    'list': list,
    'map': map,
    'max': max,
    'min': min,
    'pow': pow,
    'print': print,
    'range': range,
    'reversed': reversed,
    'round': round,
    'set': set,
    'slice': slice,
    'sorted': sorted,
    'str': str,
    'sum': sum,
    'tuple': tuple,
    'type': type,
    'zip': zip,
    'True': True,
    'False': False,
    'None': None,
    '__import__': _safe_import,  # Allow controlled imports
}


def create_sandbox_server(
    name: str = "python-sandbox-mcp",
    timeout_seconds: int = 30,
) -> FastMCP:
    """
    Create the Python Sandbox MCP server.

    Args:
        name: Server name
        timeout_seconds: Default execution timeout

    Returns:
        Configured FastMCP server
    """
    mcp = FastMCP(name)

    _timeout = timeout_seconds

    def create_execution_namespace() -> dict[str, Any]:
        """Create a safe namespace for code execution."""
        namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}

        # Pre-load common libraries
        try:
            import numpy as np
            namespace["np"] = np
            namespace["numpy"] = np
        except ImportError:
            pass

        try:
            import pandas as pd
            namespace["pd"] = pd
            namespace["pandas"] = pd
        except ImportError:
            pass

        try:
            from datetime import datetime, timedelta
            namespace["datetime"] = datetime
            namespace["timedelta"] = timedelta
        except ImportError:
            pass

        try:
            import math
            namespace["math"] = math
        except ImportError:
            pass

        try:
            from scipy import stats
            namespace["stats"] = stats
        except ImportError:
            pass

        return namespace

    @mcp.tool()
    def execute_python(
        code: str,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """
        Execute Python code in a sandboxed environment.

        Pre-loaded libraries: numpy (np), pandas (pd), math, datetime, scipy.stats

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds (default: 30)

        Returns:
            Execution result with stdout, stderr, return value, and defined variables
        """
        import time

        effective_timeout = timeout or _timeout
        start_time = time.time()

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        namespace = create_execution_namespace()

        result = ExecutionResult(success=False)

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute the code
                exec(code, namespace)

            result.success = True
            result.stdout = stdout_capture.getvalue()
            result.stderr = stderr_capture.getvalue()

            # Extract user-defined variables (exclude builtins and imports)
            exclude_keys = {"__builtins__", "np", "numpy", "pd", "pandas", "math", "datetime", "timedelta", "stats"}
            result.variables = {
                k: _serialize_value(v)
                for k, v in namespace.items()
                if k not in exclude_keys and not k.startswith("_")
            }

            # Try to get a return value from the last expression
            if result.variables:
                last_key = list(result.variables.keys())[-1]
                result.return_value = result.variables[last_key]

        except SyntaxError as e:
            result.error_type = "SyntaxError"
            result.error_message = str(e)
            result.stderr = stderr_capture.getvalue() + f"\nSyntaxError: {e}"

        except Exception as e:
            result.error_type = type(e).__name__
            result.error_message = str(e)
            result.stderr = stderr_capture.getvalue() + f"\n{traceback.format_exc()}"

        finally:
            result.execution_time_ms = int((time.time() - start_time) * 1000)

        return result.model_dump()

    @mcp.tool()
    def calculate_financial_metric(
        metric: str,
        values: dict[str, float],
    ) -> dict[str, Any]:
        """
        Calculate a specific financial metric.

        Available metrics:
        - gross_margin: (revenue - cogs) / revenue
        - operating_margin: operating_income / revenue
        - net_margin: net_income / revenue
        - roe: net_income / equity
        - roa: net_income / assets
        - current_ratio: current_assets / current_liabilities
        - debt_to_equity: total_debt / equity
        - pe_ratio: price / eps
        - ev_to_ebitda: enterprise_value / ebitda

        Args:
            metric: Name of the metric to calculate
            values: Dictionary of input values needed for calculation

        Returns:
            Calculated metric value and formula used
        """
        formulas = {
            "gross_margin": ("(revenue - cogs) / revenue", ["revenue", "cogs"]),
            "operating_margin": ("operating_income / revenue", ["operating_income", "revenue"]),
            "net_margin": ("net_income / revenue", ["net_income", "revenue"]),
            "roe": ("net_income / equity", ["net_income", "equity"]),
            "roa": ("net_income / assets", ["net_income", "assets"]),
            "current_ratio": ("current_assets / current_liabilities", ["current_assets", "current_liabilities"]),
            "debt_to_equity": ("total_debt / equity", ["total_debt", "equity"]),
            "pe_ratio": ("price / eps", ["price", "eps"]),
            "ev_to_ebitda": ("enterprise_value / ebitda", ["enterprise_value", "ebitda"]),
            "yoy_growth": ("(current - previous) / previous", ["current", "previous"]),
        }

        if metric not in formulas:
            return {
                "error": f"Unknown metric: {metric}",
                "available_metrics": list(formulas.keys()),
            }

        formula, required = formulas[metric]

        # Check required values
        missing = [k for k in required if k not in values]
        if missing:
            return {
                "error": f"Missing required values: {missing}",
                "required": required,
                "provided": list(values.keys()),
            }

        # Calculate
        try:
            result = eval(formula, {"__builtins__": {}}, values)
            return {
                "metric": metric,
                "value": result,
                "formula": formula,
                "inputs": {k: values[k] for k in required},
            }
        except ZeroDivisionError:
            return {"error": "Division by zero", "metric": metric}
        except Exception as e:
            return {"error": str(e), "metric": metric}

    @mcp.tool()
    def analyze_time_series(
        data: list[float],
        operations: list[str],
    ) -> dict[str, Any]:
        """
        Perform time series analysis on numerical data.

        Available operations:
        - mean, median, std, var
        - min, max, range
        - pct_change, cumsum
        - rolling_mean_5, rolling_mean_10
        - trend (linear regression slope)

        Args:
            data: List of numerical values (time series)
            operations: List of operations to perform

        Returns:
            Results for each requested operation
        """
        try:
            import numpy as np
        except ImportError:
            return {"error": "numpy not available"}

        arr = np.array(data)
        results: dict[str, Any] = {"data_points": len(data)}

        for op in operations:
            try:
                if op == "mean":
                    results[op] = float(np.mean(arr))
                elif op == "median":
                    results[op] = float(np.median(arr))
                elif op == "std":
                    results[op] = float(np.std(arr))
                elif op == "var":
                    results[op] = float(np.var(arr))
                elif op == "min":
                    results[op] = float(np.min(arr))
                elif op == "max":
                    results[op] = float(np.max(arr))
                elif op == "range":
                    results[op] = float(np.max(arr) - np.min(arr))
                elif op == "sum":
                    results[op] = float(np.sum(arr))
                elif op == "pct_change":
                    if len(arr) > 1:
                        pct = np.diff(arr) / arr[:-1] * 100
                        results[op] = [float(x) for x in pct]
                elif op == "cumsum":
                    results[op] = [float(x) for x in np.cumsum(arr)]
                elif op.startswith("rolling_mean_"):
                    window = int(op.split("_")[-1])
                    if len(arr) >= window:
                        rolling = np.convolve(arr, np.ones(window)/window, mode='valid')
                        results[op] = [float(x) for x in rolling]
                elif op == "trend":
                    x = np.arange(len(arr))
                    slope, intercept = np.polyfit(x, arr, 1)
                    results[op] = {"slope": float(slope), "intercept": float(intercept)}
                else:
                    results[op] = {"error": f"Unknown operation: {op}"}

            except Exception as e:
                results[op] = {"error": str(e)}

        return results

    @mcp.resource("sandbox://help")
    def help_resource() -> str:
        """Get help on using the sandbox."""
        return """
Python Sandbox MCP Server

Execute Python code with pre-loaded financial libraries.

Pre-loaded libraries:
- numpy (as np)
- pandas (as pd)
- math
- datetime, timedelta
- scipy.stats (as stats)

Example usage:
```python
# Calculate compound annual growth rate
initial_value = 1000
final_value = 1500
years = 3
cagr = (final_value / initial_value) ** (1/years) - 1
print(f"CAGR: {cagr:.2%}")
```

Available tools:
- execute_python: Run arbitrary Python code
- calculate_financial_metric: Calculate specific financial metrics
- analyze_time_series: Perform statistical analysis on time series data
""".strip()

    return mcp


def _serialize_value(value: Any) -> Any:
    """Serialize a value for JSON output."""
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
    except ImportError:
        pass

    try:
        import pandas as pd
        if isinstance(value, pd.DataFrame):
            return value.to_dict()
        if isinstance(value, pd.Series):
            return value.to_dict()
    except ImportError:
        pass

    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}

    return str(value)


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Python Sandbox MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP transports")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP transports")

    args = parser.parse_args()

    server = create_sandbox_server()

    transport_kwargs = {}
    if args.transport != "stdio":
        transport_kwargs = {"host": args.host, "port": args.port}

    server.run(transport=args.transport, **transport_kwargs)
