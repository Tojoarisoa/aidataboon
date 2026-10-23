# =============================================================================
# AIDataBoon — Cloudflare Infrastructure (Primary, Cost-Optimized)
#
# This is the PRIMARY production deployment path (~$0/month).
# It pairs Cloudflare's free-tier edge platform with two external services
# that are provisioned out-of-band (not by this file — see note below):
#   - Render  — runs the RAG API container (05_rag_service.py)
#   - Neon    — managed Postgres + pgvector (replaces local pgvector container)
#
# Render and Neon do not have first-class, actively-maintained Terraform
# providers, so this configuration treats them as external dependencies and
# references them via input variables (hostnames / API keys produced by
# Render's dashboard/Blueprint deploy of render.yaml and the Neon dashboard
# or `neonctl`). Everything *in* this file is a Cloudflare-provider resource.
#
# For teams that need full AWS instead (existing infra, compliance, VPC
# peering, etc.), see aws/08_terraform_aws.tf (~$40/month). It lives in its
# own subdirectory because it's a separate Terraform root module (its own S3
# backend, "environment"/"project_name" variables, etc.) — keeping both
# configs in this directory caused `terraform init` to fail with duplicate
# variable/provider declarations.
#
# Estimated monthly cost — Cloudflare side (free plan):
#   R2 storage, 6 buckets, <10GB total:        $0.00  (10GB free, then $0.015/GB)
#   R2 Class A/B operations:                   $0.00  (1M / 10M free per month)
#   R2 egress to the internet:                 $0.00  (always free — no egress fees)
#   Cloudflare Pages (static frontend):        $0.00  (unlimited requests, free plan)
#   Cloudflare Workers (API gateway):          $0.00  (100K requests/day free)
#   Cloudflare Tunnel (Grafana dashboard):     $0.00  (free, no public IP needed)
#   DNS, cache rules, rate limiting:           $0.00  (included on free plan)
#   Cloudflare-side total:                     ~$0/month
#
#   + Render  (free web service, RAG API, sleeps when idle): $0/month (no card required)
#   + Neon    (0.5GB storage, autosuspend):                   $0/month (free tier)
#   Grand total:                                              ~$0/month
#
# Run: terraform init && terraform apply
# Required variables (set via TF_VAR_* env vars or a .tfvars file):
#   cloudflare_api_token, cloudflare_account_id, cloudflare_zone_id,
#   upstream_api_key, tunnel_secret
#
# cloudflare_api_token permission groups (Create Custom Token in the
# dashboard) -- missing any of these fails with the generic
# "Authentication error (10000)" on the specific resource that needs it:
#   Zone   - DNS              - Edit
#   Zone   - Workers Routes   - Edit
#   Zone   - Zone WAF         - Edit  (zone-level rate-limiting ruleset)
#   Zone   - Cache Rules      - Edit  (cloudflare_ruleset.cache_data_bucket)
#   Account - Workers Scripts - Edit
#   Account - Cloudflare Pages - Edit
#   Account - R2              - Edit
#   Account - Cloudflare Tunnel - Edit
# =============================================================================

# ---------------------------------------------------------------------------
# Provider + Backend
# ---------------------------------------------------------------------------

terraform {
  required_version = ">= 1.7"

  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
  }

  # Optional remote state, stored in the "aidataboon-tfstate" R2 bucket via
  # R2's S3-compatible API. R2 buckets created *by this config* can't also
  # host its own first-apply state, so bootstrap that one bucket manually
  # (Cloudflare dashboard or `aws s3 mb --endpoint-url ...`), then uncomment:
  #
  # backend "s3" {
  #   bucket                      = "aidataboon-tfstate"
  #   key                         = "infra/cloudflare.tfstate"
  #   region                      = "auto"
  #   skip_credentials_validation = true
  #   skip_region_validation      = true
  #   skip_requesting_account_id  = true
  #   skip_metadata_api_check     = true
  #   use_path_style              = true
  #   endpoints = {
  #     s3 = "https://<cloudflare_account_id>.r2.cloudflarestorage.com"
  #   }
  # }
}

provider "cloudflare" {
  api_token = var.cloudflare_api_token
}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

variable "cloudflare_api_token" {
  description = "Cloudflare API token (needs Zone:DNS/Workers Routes/Zone WAF:Edit, Account:R2/Workers Scripts/Pages/Cloudflare Tunnel:Edit)"
  type        = string
  sensitive   = true
}

variable "cloudflare_account_id" {
  description = "Cloudflare account ID (visible in the dashboard URL: dash.cloudflare.com/<account_id>)"
  type        = string
}

variable "cloudflare_zone_id" {
  description = "Cloudflare zone ID for the domain_name zone"
  type        = string
}

variable "domain_name" {
  description = "Root domain managed by this configuration"
  type        = string
  default     = "aidataboon.com"
}

variable "environment" {
  description = "Deployment environment: dev | staging | production"
  type        = string
  default     = "production"

  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "environment must be dev, staging, or production."
  }
}

variable "project_name" {
  description = "Project identifier used for resource naming"
  type        = string
  default     = "aidataboon"
}

variable "pages_project_name" {
  description = "Cloudflare Pages project name for the static frontend"
  type        = string
  default     = "aidataboon"
}

# Render and Neon are provisioned out-of-band (Render Blueprint deploy of
# render.yaml, Neon dashboard / neonctl) — see header comment. These
# variables wire their already-provisioned hostnames and credentials into
# the Cloudflare layer.
variable "render_api_hostname" {
  description = "Render service hostname for the RAG API, e.g. <service-name>.onrender.com"
  type        = string
  default     = "aidataboon-api.onrender.com"
}

variable "upstream_api_key" {
  description = "Shared-secret API key the Worker injects as a Bearer token when proxying to the Render RAG API"
  type        = string
  sensitive   = true
}

variable "tunnel_secret" {
  description = "Base64-encoded 32-byte secret for the Grafana Cloudflare Tunnel (generate with: openssl rand -base64 32)"
  type        = string
  sensitive   = true
}

variable "r2_location" {
  description = "R2 bucket location hint: apac | eeur | enam | weur | wnam (null = automatic placement)"
  type        = string
  default     = null
}

variable "rate_limit_requests_per_10s" {
  description = "Requests allowed per 10-second window per source IP, shared by the API and agents hostnames (100 = 10 req/s; free plan allows only 1 rule per http_ratelimit phase, so both hostnames share this single limit)"
  type        = number
  default     = 100
}

# A Cloudflare zone has exactly one ruleset per phase, so the
# "http_ratelimit" phase entry point is shared across both this file and
# agents/07_cloudflare_agents.tf. This file owns it (see
# cloudflare_ruleset.rate_limit_api below); the agents config no longer
# defines its own http_ratelimit ruleset.
variable "agents_subdomain" {
  description = "Subdomain hosting the agent API (must match agents_subdomain in 07_cloudflare_agents.tf) — included in the shared rate-limit ruleset"
  type        = string
  default     = "agents"
}

variable "worker_script_name" {
  description = "Name of the Cloudflare Worker that proxies API requests to Render"
  type        = string
  default     = "aidataboon-api-gateway"
}

# ---------------------------------------------------------------------------
# Storage — R2 Buckets (S3-compatible, zero egress fees)
# ---------------------------------------------------------------------------

# Mirrors the bucket layout used locally by 01_local_stack.yml's MinIO and
# referenced by 03_lakehouse.py's LakehouseConfig (bronze/bronze-delta/
# silver/gold), 07_observability.py (mlflow-artifacts) and Airflow remote
# logging (airflow-logs).
locals {
  r2_buckets = [
    "aidataboon-bronze",
    "aidataboon-bronze-delta",
    "aidataboon-silver",
    "aidataboon-gold",
    "mlflow-artifacts",
    "airflow-logs",
  ]
}

# Cost: first 10GB/month free, then $0.015/GB-month storage. Class A
# (write/list) ops: 1M/month free, then $4.50/million. Class B (read) ops:
# 10M/month free, then $0.36/million. Egress is always free.
resource "cloudflare_r2_bucket" "buckets" {
  for_each = toset(local.r2_buckets)

  account_id = var.cloudflare_account_id
  name       = each.value
  location   = var.r2_location
}

# Public read access for the Gold layer, mirroring `mc anonymous set
# download local/aidataboon-gold` in 01_local_stack.yml. A custom domain
# binding is what makes objects reachable at https://data.aidataboon.com/...
# without signed URLs.
#
# NOT managed here: the cloudflare/cloudflare ~> 4.0 provider has no
# resource for R2 custom domains (that API is only exposed via
# cloudflare_r2_custom_domain in provider v5, a breaking major upgrade from
# the rest of this file). Bind it manually once: R2 -> aidataboon-gold ->
# Settings -> Custom Domains -> add "data.aidataboon.com" (Cloudflare
# provisions the DNS record automatically — do not also define a
# cloudflare_record for "data" below).

# ---------------------------------------------------------------------------
# Frontend — Cloudflare Pages
# ---------------------------------------------------------------------------

# Direct-upload project (no git integration / build_config needed): the
# Next.js frontend is built and deployed by 10_cicd_pipeline.yml via
# `cloudflare/pages-action@v1`, which uploads the `frontend/out` directory.
# Cost: unlimited requests/bandwidth on the free plan; 500 builds/month if
# CI-driven builds are used instead of direct upload.
resource "cloudflare_pages_project" "frontend" {
  account_id        = var.cloudflare_account_id
  name              = var.pages_project_name
  production_branch = "main"
}

# DNS alone (cloudflare_record.root below) is not enough for Pages to serve
# the apex domain — the hostname must also be registered as a custom domain
# on the Pages project, or requests to it return 522.
resource "cloudflare_pages_domain" "root" {
  account_id   = var.cloudflare_account_id
  project_name = cloudflare_pages_project.frontend.name
  domain       = var.domain_name
}

# ---------------------------------------------------------------------------
# API Gateway — Cloudflare Worker (proxies api.aidataboon.com -> Render)
# ---------------------------------------------------------------------------

# Cost: 100K requests/day free, then $0.30/million. Handles CORS, auth header
# injection, /health short-circuiting, and routes /v1/* to the Render RAG
# API. Per-IP rate limiting is enforced ahead of the Worker by the
# "rate_limit_api" ruleset below (cheaper and runs before Worker invocation).
resource "cloudflare_worker_script" "api_gateway" {
  account_id = var.cloudflare_account_id
  name       = var.worker_script_name
  module     = true

  content = <<-JS
    // AIDataBoon API Gateway — proxies api.${var.domain_name}/v1/* to the
    // Render RAG API and answers /health directly. Per-IP rate limiting
    // (10 req/s) is enforced by the "rate_limit_api" cloudflare_ruleset,
    // which runs ahead of this Worker.

    const ALLOWED_ORIGIN = "https://${var.domain_name}";
    const UPSTREAM_HOST = "${var.render_api_hostname}";
    const UPSTREAM_BASE = "https://" + UPSTREAM_HOST;

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

        if (url.pathname === "/health") {
          const body = JSON.stringify({ status: "ok", proxy: "cloudflare-worker" });
          return new Response(body, {
            headers: Object.assign({ "content-type": "application/json" }, corsHeaders()),
          });
        }

        if (url.pathname.indexOf("/v1/") === 0) {
          const upstreamUrl = UPSTREAM_BASE + url.pathname.slice(3) + url.search;
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
        }

        return new Response("Not found", { status: 404, headers: corsHeaders() });
      },
    };
  JS
}

resource "cloudflare_worker_secret" "upstream_api_key" {
  account_id  = var.cloudflare_account_id
  script_name = cloudflare_worker_script.api_gateway.name
  name        = "UPSTREAM_API_KEY"
  secret_text = var.upstream_api_key
}

resource "cloudflare_worker_route" "api_gateway" {
  zone_id     = var.cloudflare_zone_id
  pattern     = "api.${var.domain_name}/*"
  script_name = cloudflare_worker_script.api_gateway.name
}

# ---------------------------------------------------------------------------
# Dashboard — Cloudflare Tunnel (exposes local Grafana, no public IP)
# ---------------------------------------------------------------------------

# Cost: free. The cloudflared daemon must be run alongside Grafana (e.g. as
# an additional service in 01_local_stack.yml or on the production host)
# using this tunnel's credentials: `cloudflared tunnel run --token <token>`.
# Grafana's own authentication (see 01_local_stack.yml) remains the access
# control layer; add Cloudflare Access in front of the hostname for SSO/MFA
# if needed.
resource "cloudflare_tunnel" "grafana" {
  account_id = var.cloudflare_account_id
  name       = "${var.project_name}-grafana"
  secret     = var.tunnel_secret
}

resource "cloudflare_tunnel_config" "grafana" {
  account_id = var.cloudflare_account_id
  tunnel_id  = cloudflare_tunnel.grafana.id

  config {
    ingress_rule {
      hostname = "dashboard.${var.domain_name}"
      service  = "http://localhost:3000"
    }

    # Catch-all rule is required as the last ingress rule.
    ingress_rule {
      service = "http_status:404"
    }
  }
}

# ---------------------------------------------------------------------------
# DNS Records
# ---------------------------------------------------------------------------

# Root domain -> Cloudflare Pages (static frontend)
resource "cloudflare_record" "root" {
  zone_id = var.cloudflare_zone_id
  name    = "@"
  type    = "CNAME"
  content = cloudflare_pages_project.frontend.subdomain
  proxied = true
}

# api.aidataboon.com -> Render. The cloudflare_worker_route above intercepts
# requests to this hostname at the edge, so this origin is rarely hit
# directly — it exists so the hostname resolves and is proxied (orange
# cloud) for the Worker route to match against.
resource "cloudflare_record" "api" {
  zone_id = var.cloudflare_zone_id
  name    = "api"
  type    = "CNAME"
  content = var.render_api_hostname
  proxied = true
}

# dashboard.aidataboon.com -> Cloudflare Tunnel -> Grafana:3000
resource "cloudflare_record" "dashboard" {
  zone_id = var.cloudflare_zone_id
  name    = "dashboard"
  type    = "CNAME"
  content = cloudflare_tunnel.grafana.cname
  proxied = true
}

# data.aidataboon.com is intentionally NOT defined here — it is managed by
# the cloudflare_r2_custom_domain.gold_public resource above.

# ---------------------------------------------------------------------------
# Edge Rules — Cache + Rate Limiting (zone-level rulesets, no Worker cost)
# ---------------------------------------------------------------------------

# Cache Gold-layer data served from R2 aggressively at the edge — it's
# immutable per pipeline run and safe to cache for an hour.
resource "cloudflare_ruleset" "cache_data_bucket" {
  zone_id     = var.cloudflare_zone_id
  name        = "${var.project_name}-cache-rules"
  description = "Cache data.${var.domain_name} responses at the edge"
  kind        = "zone"
  phase       = "http_request_cache_settings"

  rules {
    description = "Cache data subdomain (Gold layer R2 bucket) for 1 hour"
    expression  = "(http.host eq \"data.${var.domain_name}\")"
    action      = "set_cache_settings"

    action_parameters {
      cache = true

      edge_ttl {
        mode    = "override_origin"
        default = 3600
      }

      browser_ttl {
        mode    = "override_origin"
        default = 3600
      }
    }
  }
}

# Per-IP rate limiting for the RAG API and the agent API, enforced before
# the Workers run (cheaper than rate limiting inside Worker code). A
# Cloudflare zone only has a single ruleset for the "http_ratelimit" phase,
# so this one resource covers both api.aidataboon.com (this file) and
# agents.aidataboon.com (agents/07_cloudflare_agents.tf).
# The free plan also caps the http_ratelimit phase at 1 rule, so both
# hostnames share a single rule/limit rather than getting independent
# thresholds.
# Default: 100 requests per 10-second window per source IP == 10 req/s,
# matching 00_project_design.md section 7.3.
resource "cloudflare_ruleset" "rate_limit_api" {
  zone_id     = var.cloudflare_zone_id
  name        = "${var.project_name}-rate-limiting"
  description = "Rate-limit api.${var.domain_name} and ${var.agents_subdomain}.${var.domain_name} per source IP"
  kind        = "zone"
  phase       = "http_ratelimit"

  rules {
    description = "Block IPs exceeding the API rate limit"
    expression  = "(http.host eq \"api.${var.domain_name}\") or (http.host eq \"${var.agents_subdomain}.${var.domain_name}\")"
    action      = "block"

    ratelimit {
      characteristics     = ["ip.src", "cf.colo.id"]
      period              = 10
      requests_per_period = var.rate_limit_requests_per_10s
      mitigation_timeout  = 10
    }

    action_parameters {
      response {
        status_code  = 429
        content_type = "application/json"
        content      = "{\"error\": \"rate limit exceeded\"}"
      }
    }
  }
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "pages_project_subdomain" {
  description = "Default *.pages.dev hostname for the frontend (root DNS record CNAMEs to this)"
  value       = cloudflare_pages_project.frontend.subdomain
}

output "worker_route_pattern" {
  description = "Route pattern the API gateway Worker is bound to"
  value       = cloudflare_worker_route.api_gateway.pattern
}

output "r2_bucket_names" {
  description = "Names of all provisioned R2 buckets"
  value       = [for b in cloudflare_r2_bucket.buckets : b.name]
}

output "tunnel_id" {
  description = "Cloudflare Tunnel ID for Grafana — run `cloudflared tunnel run --token <token>` for this tunnel to bring the dashboard online"
  value       = cloudflare_tunnel.grafana.id
}

output "tunnel_cname" {
  description = "CNAME target Cloudflare assigned to the Grafana tunnel (matches the dashboard DNS record)"
  value       = cloudflare_tunnel.grafana.cname
}
