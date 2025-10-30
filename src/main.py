"""FastAPI server for Verisage - Multi-LLM Oracle."""

from fastapi import FastAPI, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import settings
from src.models import OracleQuery, OracleResult
from src.oracle import oracle

# Custom CSS for API docs
CUSTOM_SWAGGER_CSS = """
body { background: #0a0a0a; }
.swagger-ui { font-family: 'Inter', sans-serif; }
.swagger-ui .topbar { background: #111; border-bottom: 1px solid #222; }
.swagger-ui .topbar .download-url-wrapper { display: none; }
.swagger-ui .info { background: #111; border-bottom: 1px solid #222; margin: 0; padding: 40px; }
.swagger-ui .info .title { color: #e4e4e7; font-family: 'Space Grotesk', sans-serif; font-size: 3em; font-weight: 600; letter-spacing: -0.03em; }
.swagger-ui .info .description p { color: #71717a; font-size: 0.95em; }
.swagger-ui .scheme-container { background: #111; border: none; padding: 20px 40px; border-bottom: 1px solid #222; }
.swagger-ui .opblock-tag { border-bottom: 1px solid #222; color: #a1a1aa; background: transparent; }
.swagger-ui .opblock { background: #111; border: 1px solid #222; border-radius: 0; margin: 0 0 1px 0; }
.swagger-ui .opblock .opblock-summary { background: transparent; border: none; padding: 20px; }
.swagger-ui .opblock.opblock-post { border-left: 3px solid #10b981; }
.swagger-ui .opblock.opblock-get { border-left: 3px solid #10b981; }
.swagger-ui .opblock .opblock-summary-method { background: #10b981; color: #0a0a0a; border-radius: 0; font-weight: 500; }
.swagger-ui .opblock .opblock-summary-path { color: #e4e4e7; }
.swagger-ui .opblock .opblock-summary-description { color: #71717a; }
.swagger-ui .opblock .opblock-section-header { background: #0a0a0a; border-bottom: 1px solid #222; }
.swagger-ui .opblock .opblock-section-header h4 { color: #a1a1aa; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.05em; }
.swagger-ui .opblock-body { background: #0a0a0a; color: #d4d4d8; }
.swagger-ui .parameters-col_description { color: #71717a; }
.swagger-ui .parameter__name { color: #e4e4e7; }
.swagger-ui .parameter__type { color: #10b981; }
.swagger-ui .response-col_status { color: #10b981; }
.swagger-ui .response-col_description { color: #71717a; }
.swagger-ui .btn { border-radius: 0; font-weight: 500; }
.swagger-ui .btn.execute { background: #e4e4e7; color: #0a0a0a; border: none; }
.swagger-ui .btn.execute:hover { background: #fafafa; }
.swagger-ui .btn.cancel { background: transparent; color: #71717a; border: 1px solid #27272a; }
.swagger-ui .btn.cancel:hover { border-color: #10b981; color: #10b981; }
.swagger-ui textarea { background: #0a0a0a; border: 1px solid #27272a; color: #e4e4e7; border-radius: 0; }
.swagger-ui input { background: #0a0a0a; border: 1px solid #27272a; color: #e4e4e7; border-radius: 0; }
.swagger-ui select { background: #0a0a0a; border: 1px solid #27272a; color: #e4e4e7; border-radius: 0; }
.swagger-ui .responses-inner { background: #0a0a0a; }
.swagger-ui .model-box { background: #0a0a0a; border: 1px solid #222; border-radius: 0; }
.swagger-ui .model { color: #d4d4d8; }
.swagger-ui .model-title { color: #e4e4e7; }
.swagger-ui .prop-type { color: #10b981; }
.swagger-ui .prop-format { color: #71717a; }
.swagger-ui table thead tr th { color: #a1a1aa; border-bottom: 1px solid #222; }
.swagger-ui table tbody tr td { color: #d4d4d8; border-bottom: 1px solid #222; }
.swagger-ui .tab li { color: #71717a; }
.swagger-ui .tab li.active { color: #e4e4e7; }
.swagger-ui .markdown p, .swagger-ui .markdown code { color: #d4d4d8; }
"""

# Initialize FastAPI app
app = FastAPI(
    title="Verisage",
    description="Multi-LLM oracle for dispute resolution and fact verification",
    version="0.1.0",
    docs_url=None,  # Disable default docs
    redoc_url=None,
    swagger_ui_parameters={
        "syntaxHighlight.theme": "nord",
        "defaultModelsExpandDepth": 1,
    },
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Serve custom styled API documentation."""
    html = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - API Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )

    # Inject custom CSS into the HTML
    html_str = html.body.decode()
    html_str = html_str.replace(
        "</head>",
        f"<style>{CUSTOM_SWAGGER_CSS}</style></head>"
    )

    return HTMLResponse(content=html_str)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/query", response_model=OracleResult)
async def query_oracle(query: OracleQuery) -> OracleResult:
    """Query the oracle with a dispute question.

    Args:
        query: The dispute question to resolve

    Returns:
        OracleResult with aggregated decision from all LLM backends
    """
    result = await oracle.resolve_dispute(query.query)
    return result


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
