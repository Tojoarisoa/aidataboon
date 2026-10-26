# =============================================================================
# AIDataBoon Agents: Cloudflare Infrastructure (agents.aidataboon.com)
#
# Implements ADR-7 (00_project_design.md): a separate Terraform root module
# (own state) from data_platform/09_cloudflare_infra.tf, adding
# agents.aidataboon.com on top of the same zone. Follows the same "$0/mo"
# pattern: a Cloudflare Worker proxies to the agent API container deployed
# out-of-band (Render, same as the RAG API), fronted by zone-level rate
# limiting (no extra Worker invocation cost for blocked requests).
#
# Run: terraform init && terraform apply
# Required variables (set via TF_VAR_* env vars or a .tfvars file):
#   cloudflare_api_token, cloudflare_account_id, cloudflare_zone_id,
#   upstream_api_key
#
# cloudflare_api_token permission groups (Create Custom Token in the
# dashboard) -- missing any of these fails with the generic
# "Authentication error (10000)" on the specific resource that needs it:
#   Zone   - DNS              - Edit
#   Zone   - Workers Routes   - Edit
#   Zone   - Zone WAF         - Edit  (zone-level rulesets / rate limiting)
#   Account - Workers Scripts - Edit
#
# Estimated incremental monthly cost:
#   Cloudflare Worker (agents gateway):  $0.00  (100K requests/day free)
#   DNS record + rate-limit ruleset:     $0.00  (included on free plan)
#   + Render (free web service, agent-api, sleeps when idle): $0.00 (no card required)
#   Total:                                ~$0/month
# =============================================================================

terraform {
  required_version = ">= 1.7"

  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
  }
}

provider "cloudflare" {
  api_token = var.cloudflare_api_token
}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

variable "cloudflare_api_token" {
  description = "Cloudflare API token (needs Zone:DNS/Workers Routes/Zone WAF:Edit, Account:Workers Scripts:Edit)"
  type        = string
  sensitive   = true
}

variable "cloudflare_account_id" {
  description = "Cloudflare account ID (visible in the dashboard URL: dash.cloudflare.com/<account_id>)"
  type        = string
}

variable "cloudflare_zone_id" {
  description = "Cloudflare zone ID for the domain_name zone (same zone as 09_cloudflare_infra.tf)"
  type        = string
}

variable "domain_name" {
  description = "Root domain managed by this configuration"
  type        = string
  default     = "aidataboon.com"
}

variable "agents_subdomain" {
  description = "Subdomain hosting the agent API"
  type        = string
  default     = "agents"
}

variable "project_name" {
  description = "Project identifier used for resource naming"
  type        = string
  default     = "aidataboon-agents"
}

# 06_agent_api.py is provisioned out-of-band (Render Blueprint deploy of
# render.yaml from agents/, same pattern as the RAG API in
# 09_cloudflare_infra.tf). This variable wires its hostname into the
# Cloudflare layer.
variable "render_agent_hostname" {
  description = "Render service hostname for the agent API, e.g. <service-name>.onrender.com"
  type        = string
  default     = "aidataboon-agents.onrender.com"
}

variable "upstream_api_key" {
  description = "Shared-secret API key the Worker injects as a Bearer token when proxying to the Render agent API"
  type        = string
  sensitive   = true
}

variable "worker_script_name" {
  description = "Name of the Cloudflare Worker that proxies agent API requests to Render"
  type        = string
  default     = "aidataboon-agents-gateway"
}

# ---------------------------------------------------------------------------
# API Gateway: Cloudflare Worker (proxies agents.aidataboon.com -> Render)
# ---------------------------------------------------------------------------

# Cost: 100K requests/day free, then $0.30/million. Proxies /agents/*,
# /prompts/*, /health, and /metrics to the Render agent API. CORS is scoped
# to the main site origin since /agents/chat is called from the frontend.
resource "cloudflare_worker_script" "agents_gateway" {
  account_id = var.cloudflare_account_id
  name       = var.worker_script_name
  module     = true

  content = <<-JS
    // AIDataBoon Agents Gateway: proxies ${var.agents_subdomain}.${var.domain_name}/*
    // to the Render agent API (06_agent_api.py). Per-IP rate limiting is
    // enforced by the shared "rate_limit_api" cloudflare_ruleset in
    // data_platform/09_cloudflare_infra.tf, which runs ahead of this
    // Worker.

    const ALLOWED_ORIGIN = "https://${var.domain_name}";
    const UPSTREAM_HOST = "${var.render_agent_hostname}";
    const UPSTREAM_BASE = "https://" + UPSTREAM_HOST;
    const PASSTHROUGH_PREFIXES = ["/agents/", "/prompts/", "/health", "/metrics"];

    function corsHeaders() {
      return {
        "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Max-Age": "86400",
      };
    }

    export default {
      async fetch(request, env) {
        const url = new URL(request.url);

        if (request.method === "OPTIONS") {
          return new Response(null, { status: 204, headers: corsHeaders() });
        }

        const allowed = PASSTHROUGH_PREFIXES.some((p) => url.pathname.indexOf(p) === 0);
        if (!allowed) {
          return new Response("Not found", { status: 404, headers: corsHeaders() });
        }

        const upstreamUrl = UPSTREAM_BASE + url.pathname + url.search;
        const upstreamRequest = new Request(upstreamUrl, request);
        upstreamRequest.headers.set("Authorization", "Bearer " + env.UPSTREAM_API_KEY);
        upstreamRequest.headers.set("Host", UPSTREAM_HOST);

        const response = await fetch(upstreamRequest);
        const proxied = new Response(response.body, response);
        const cors = corsHeaders();
        for (const key in cors) {
          proxied.headers.set(key, cors[key]);
        }
        return proxied;
      },
    };
  JS
}

resource "cloudflare_worker_secret" "upstream_api_key" {
  account_id  = var.cloudflare_account_id
  script_name = cloudflare_worker_script.agents_gateway.name
  name        = "UPSTREAM_API_KEY"
  secret_text = var.upstream_api_key
}

resource "cloudflare_worker_route" "agents_gateway" {
  zone_id     = var.cloudflare_zone_id
  pattern     = "${var.agents_subdomain}.${var.domain_name}/*"
  script_name = cloudflare_worker_script.agents_gateway.name
}

# ---------------------------------------------------------------------------
# DNS Record
# ---------------------------------------------------------------------------

# agents.aidataboon.com -> Render. The cloudflare_worker_route above
# intercepts requests at the edge; this record exists so the hostname
# resolves and is proxied (orange cloud) for the Worker route to match.
resource "cloudflare_record" "agents" {
  zone_id = var.cloudflare_zone_id
  name    = var.agents_subdomain
  type    = "CNAME"
  content = var.render_agent_hostname
  proxied = true
}

# ---------------------------------------------------------------------------
# Edge Rules: Rate Limiting
# ---------------------------------------------------------------------------

# Per-IP rate limiting for the agent API is defined in
# cloudflare_ruleset.rate_limit_api in data_platform/09_cloudflare_infra.tf
# (not here): a Cloudflare zone has exactly one ruleset per phase, and
# 09_cloudflare_infra.tf already owns the "http_ratelimit" phase for
# api.${var.domain_name}. Its rate_limit_api ruleset includes a second rule
# for ${var.agents_subdomain}.${var.domain_name}, configurable via that
# file's agents_rate_limit_requests_per_10s variable.

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "worker_route_pattern" {
  description = "Route pattern the agents gateway Worker is bound to"
  value       = cloudflare_worker_route.agents_gateway.pattern
}

output "agents_hostname" {
  description = "Public hostname for the agent API"
  value       = "${var.agents_subdomain}.${var.domain_name}"
}
