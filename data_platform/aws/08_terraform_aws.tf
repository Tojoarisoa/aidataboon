# =============================================================================
# AIDataBoon: AWS Infrastructure (Cost-Optimized)
# Optional alternative to 09_cloudflare_infra.tf
#
# IMPORTANT: This is the OPTIONAL AWS deployment path.
# For most use cases, prefer 09_cloudflare_infra.tf which costs ~$0-5/month.
# This file is for teams that need full AWS (existing infra, compliance, etc.)
#
# Estimated monthly cost (minimal config):
#   ECS Fargate Spot (0.5 vCPU/1GB, ~50% uptime): ~$8/month
#   RDS db.t3.micro (PostgreSQL):                  ~$15/month  (free tier yr 1)
#   ALB:                                           ~$16/month
#   S3 (10GB data, standard requests):             ~$0.50/month
#   CloudWatch Logs (7-day retention):             ~$1/month
#   NAT Gateway (disabled by default):             ~$32/month  SKIPPED
#   Total (no NAT):                                ~$40/month
#
# Run: terraform init && terraform apply
# Required env vars: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (or OIDC)
# =============================================================================

# ---------------------------------------------------------------------------
# Provider + Backend
# ---------------------------------------------------------------------------

terraform {
  required_version = ">= 1.7"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Remote state stored in S3 with DynamoDB locking.
  # Bootstrap once with: aws s3 mb s3://aidataboon-tfstate --region us-east-1
  #                       aws dynamodb create-table --table-name aidataboon-tfstate-lock \
  #                         --attribute-definitions AttributeName=LockID,AttributeType=S \
  #                         --key-schema AttributeName=LockID,KeyType=HASH \
  #                         --billing-mode PAY_PER_REQUEST --region us-east-1
  backend "s3" {
    bucket         = "aidataboon-tfstate"
    key            = "infra/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "aidataboon-tfstate-lock"
    encrypt        = true
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
      Repository  = "github.com/yourusername/aidataboon"
    }
  }
}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

variable "region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
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

variable "db_password" {
  description = "Master password for RDS PostgreSQL (stored in SSM after first apply)"
  type        = string
  sensitive   = true
}

variable "app_image_tag" {
  description = "Docker image tag to deploy (git SHA or 'latest')"
  type        = string
  default     = "latest"
}

# Cost flag: NAT Gateway adds ~$32/month. Disabled by default.
# Enable only if ECS tasks need outbound internet and you cannot use
# VPC endpoints or public subnets with assigned public IPs.
variable "enable_nat_gateway" {
  description = "Create NAT Gateway for private subnets; adds ~$32/month"
  type        = bool
  default     = false
}

variable "fargate_cpu" {
  description = "CPU units for RAG API Fargate task (256 = 0.25 vCPU)"
  type        = number
  default     = 512 # 0.5 vCPU, cost ~$0.02/hour at on-demand, ~$0.01 Spot
}

variable "fargate_memory" {
  description = "Memory MiB for RAG API Fargate task"
  type        = number
  default     = 1024 # 1 GB
}

# ---------------------------------------------------------------------------
# Data Sources
# ---------------------------------------------------------------------------

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

data "aws_availability_zones" "available" {
  state = "available"
}

# ---------------------------------------------------------------------------
# Networking: VPC + Public Subnets (no private subnets = no NAT cost)
# ---------------------------------------------------------------------------
# Architecture decision: We use only public subnets and assign public IPs
# directly to Fargate tasks. Traffic is secured at the security group level.
# This avoids the ~$32/month NAT Gateway cost.
# Trade-off: Tasks are directly internet-routable (mitigated by SG rules).

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = "${var.project_name}-vpc" }
}

# Public subnet A: ALB + Fargate tasks
# Cost: FREE (no charge for subnets or public IPs on EC2/Fargate)
resource "aws_subnet" "public_a" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "us-east-1a"
  map_public_ip_on_launch = true

  tags = { Name = "${var.project_name}-public-a", Tier = "public" }
}

# Public subnet B: ALB requires 2 AZs minimum
resource "aws_subnet" "public_b" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.2.0/24"
  availability_zone       = "us-east-1b"
  map_public_ip_on_launch = true

  tags = { Name = "${var.project_name}-public-b", Tier = "public" }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${var.project_name}-igw" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = { Name = "${var.project_name}-rt-public" }
}

resource "aws_route_table_association" "public_a" {
  subnet_id      = aws_subnet.public_a.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "public_b" {
  subnet_id      = aws_subnet.public_b.id
  route_table_id = aws_route_table.public.id
}

# ---------------------------------------------------------------------------
# Security Groups
# ---------------------------------------------------------------------------

# ALB: accepts HTTP (80) and HTTPS (443) from the internet
resource "aws_security_group" "alb_sg" {
  name        = "${var.project_name}-alb-sg"
  description = "Allow HTTP/HTTPS inbound to ALB"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTP from internet"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS from internet"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-alb-sg" }
}

# ECS tasks: only accepts traffic from the ALB on port 8000
resource "aws_security_group" "ecs_sg" {
  name        = "${var.project_name}-ecs-sg"
  description = "Allow port 8000 from ALB only"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "App port from ALB"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb_sg.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-ecs-sg" }
}

# RDS: only accepts PostgreSQL traffic from ECS tasks
resource "aws_security_group" "rds_sg" {
  name        = "${var.project_name}-rds-sg"
  description = "Allow PostgreSQL 5432 from ECS tasks only"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "PostgreSQL from ECS"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_sg.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-rds-sg" }
}

# ---------------------------------------------------------------------------
# Storage: S3 Buckets (cost-optimized)
# ---------------------------------------------------------------------------

# Data Lake: Bronze / Silver / Gold medallion architecture
# Cost: ~$0.023/GB/month (S3 Standard), transitions to Intelligent-Tiering after 30 days
resource "aws_s3_bucket" "data_lake" {
  bucket = "${var.project_name}-data-lake-${data.aws_caller_identity.current.account_id}"

  tags = { Name = "${var.project_name}-data-lake", Tier = "all" }
}

resource "aws_s3_bucket_versioning" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

# SSE-S3 (AES-256) is free. Avoid SSE-KMS: costs $0.03 per 10K API calls.
resource "aws_s3_bucket_server_side_encryption_configuration" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256" # SSE-S3: free | SSE-KMS: $0.03/10K API calls
    }
    bucket_key_enabled = false
  }
}

# Intelligent-Tiering: auto-moves infrequently accessed objects to cheaper tiers.
# Monitoring fee: $0.0025 per 1K objects/month for objects > 128KB.
# Breaks even vs S3 Standard at ~1 access per 30 days.
resource "aws_s3_bucket_lifecycle_configuration" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id

  rule {
    id     = "intelligent-tiering-transition"
    status = "Enabled"

    filter {
      prefix = "" # applies to all objects
    }

    transition {
      days          = 30
      storage_class = "INTELLIGENT_TIERING"
    }

    # Expire non-current versions after 90 days to control versioning storage costs
    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}

resource "aws_s3_bucket_public_access_block" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# MLflow artifact store: separate bucket for model tracking
# Cost: typically <$1/month for small teams
resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "${var.project_name}-mlflow-artifacts-${data.aws_caller_identity.current.account_id}"

  tags = { Name = "${var.project_name}-mlflow-artifacts", Purpose = "mlflow" }
}

resource "aws_s3_bucket_versioning" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Terraform state bucket: no lifecycle rules; state files are small
# Cost: negligible (<$0.01/month)
resource "aws_s3_bucket" "tfstate" {
  bucket = "aidataboon-tfstate"

  tags = { Name = "aidataboon-tfstate", Purpose = "terraform-state" }

  lifecycle {
    prevent_destroy = true # never accidentally delete state
  }
}

resource "aws_s3_bucket_versioning" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------------------------------------------------------------------------
# Database: RDS PostgreSQL 16 with pgvector
# ---------------------------------------------------------------------------
# Alternative: Neon.tech free tier saves ~$20/month (0.5 GB storage, serverless)
# Use Neon for dev/staging; RDS only when you need dedicated compute + pgvector.

resource "aws_db_parameter_group" "postgres16" {
  name        = "${var.project_name}-postgres16-pgvector"
  family      = "postgres16"
  description = "AIDataBoon PostgreSQL 16 with pgvector extension enabled"

  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements,vector" # enable pgvector
  }

  parameter {
    name  = "log_connections"
    value = "1"
  }

  parameter {
    name         = "log_min_duration_statement"
    value        = "1000" # log queries > 1s
    apply_method = "immediate"
  }

  tags = { Name = "${var.project_name}-pg16-params" }
}

resource "aws_db_subnet_group" "main" {
  name       = "${var.project_name}-db-subnet-group"
  subnet_ids = [aws_subnet.public_a.id, aws_subnet.public_b.id]

  tags = { Name = "${var.project_name}-db-subnet-group" }
}

# Cost: db.t3.micro = ~$0/month (free tier yr 1), ~$15/month after
# Multi-AZ doubles the cost: disabled for cost optimization.
# Enable multi_az = true for production-critical workloads (~$30/month).
resource "aws_db_instance" "postgres" {
  identifier        = "${var.project_name}-postgres"
  engine            = "postgres"
  engine_version    = "16.3"
  instance_class    = "db.t3.micro" # Cost: ~$15/month | Free tier: 750h/month yr 1

  db_name  = "aidataboon"
  username = "aidataboon_admin"
  password = var.db_password # stored in SSM after first apply

  # Storage: 20GB gp2 to start, auto-scales up to 100GB
  # Cost: $0.115/GB/month (gp2) = ~$2.30/month for 20GB
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp2"
  storage_encrypted     = true

  multi_az               = false # Cost saving: single-AZ only
  publicly_accessible    = false
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  parameter_group_name   = aws_db_parameter_group.postgres16.name

  backup_retention_period = 7     # 7 days of automated backups (free)
  backup_window           = "03:00-04:00"
  maintenance_window      = "Mon:04:00-Mon:05:00"

  skip_final_snapshot       = false
  final_snapshot_identifier = "${var.project_name}-postgres-final-snapshot"

  deletion_protection = true # prevent accidental deletion

  performance_insights_enabled = false # Cost: free for 7 days, $0.02/vCPU/hr beyond

  tags = { Name = "${var.project_name}-postgres" }
}

# ---------------------------------------------------------------------------
# ECR: Container Registry
# ---------------------------------------------------------------------------
# Cost: $0.10/GB/month for storage; first 500MB/month free

resource "aws_ecr_repository" "api" {
  name                 = "${var.project_name}/rag-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true # free: scans for CVEs on every push
  }

  tags = { Name = "${var.project_name}-ecr-api" }
}

# Keep only the last 10 images to control storage costs
resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = { type = "expire" }
      }
    ]
  })
}

# ---------------------------------------------------------------------------
# ECS Cluster + Task Definitions
# ---------------------------------------------------------------------------

# Container insights disabled by default; adds ~$0.30/GB of log data ingested.
# Enable in production if you need detailed metrics: containerInsights = "enabled"
resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"

  setting {
    name  = "containerInsights"
    value = "disabled" # Cost: saves ~$0.30/GB log data; enable for prod debugging
  }

  tags = { Name = "${var.project_name}-cluster" }
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name = aws_ecs_cluster.main.name

  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT" # 70% Spot
    weight            = 70
    base              = 0
  }
}

# RAG API Task Definition: 0.5 vCPU / 1 GB
# Cost: $0.01048/vCPU/hr + $0.00116/GB/hr on-demand
#       Spot saves up to 70%: ~$0.003/vCPU/hr
resource "aws_ecs_task_definition" "rag_api" {
  family                   = "${var.project_name}-rag-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.fargate_cpu    # 512 = 0.5 vCPU
  memory                   = var.fargate_memory # 1024 = 1 GB
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = "rag-api"
      image     = "${aws_ecr_repository.api.repository_url}:${var.app_image_tag}"
      essential = true

      portMappings = [
        {
          containerPort = 8000
          protocol      = "tcp"
        }
      ]

      environment = [
        { name = "ENVIRONMENT", value = var.environment },
        { name = "PORT", value = "8000" },
        { name = "WORKERS", value = "2" },
        { name = "S3_DATA_LAKE_BUCKET", value = aws_s3_bucket.data_lake.bucket },
        { name = "S3_MLFLOW_BUCKET", value = aws_s3_bucket.mlflow_artifacts.bucket },
        { name = "RDS_HOST", value = aws_db_instance.postgres.address },
        { name = "RDS_PORT", value = "5432" },
        { name = "RDS_DB", value = "aidataboon" },
        { name = "RDS_USER", value = "aidataboon_admin" },
      ]

      secrets = [
        {
          name      = "DB_PASSWORD"
          valueFrom = aws_ssm_parameter.db_password.arn
        },
        {
          name      = "REDIS_URL"
          valueFrom = aws_ssm_parameter.redis_url.arn
        },
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "rag-api"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])

  tags = { Name = "${var.project_name}-rag-api-task" }
}

# Worker Task Definition: 1 vCPU / 2 GB (heavier embedding workloads)
# Cost: ~$0.021/hr on-demand, ~$0.007/hr Spot
resource "aws_ecs_task_definition" "worker" {
  family                   = "${var.project_name}-worker"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 1024 # 1 vCPU
  memory                   = 2048 # 2 GB
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = "worker"
      image     = "${aws_ecr_repository.api.repository_url}:${var.app_image_tag}"
      essential = true
      command   = ["python", "-m", "aidataboon.worker"]

      environment = [
        { name = "ENVIRONMENT", value = var.environment },
        { name = "WORKER_MODE", value = "true" },
        { name = "S3_DATA_LAKE_BUCKET", value = aws_s3_bucket.data_lake.bucket },
      ]

      secrets = [
        {
          name      = "DB_PASSWORD"
          valueFrom = aws_ssm_parameter.db_password.arn
        },
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "worker"
        }
      }
    }
  ])

  tags = { Name = "${var.project_name}-worker-task" }
}

# ---------------------------------------------------------------------------
# ALB: Application Load Balancer
# ---------------------------------------------------------------------------
# Cost: ~$16/month (ALB fixed fee) + $0.008/LCU-hour

resource "aws_lb" "main" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = [aws_subnet.public_a.id, aws_subnet.public_b.id]

  enable_deletion_protection = true
  enable_http2               = true

  access_logs {
    bucket  = aws_s3_bucket.data_lake.bucket
    prefix  = "alb-logs"
    enabled = false # enable for debugging; adds S3 PUT costs
  }

  tags = { Name = "${var.project_name}-alb" }
}

resource "aws_lb_target_group" "api" {
  name        = "${var.project_name}-api-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip" # required for Fargate awsvpc networking

  health_check {
    enabled             = true
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    matcher             = "200"
  }

  tags = { Name = "${var.project_name}-api-tg" }
}

# HTTP → HTTPS redirect (no content served on port 80)
resource "aws_lb_listener" "http_redirect" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "redirect"

    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

# HTTPS listener using ACM certificate (Certificate must be requested separately
# or imported; ACM certs are FREE, only pay for Load Balancer usage)
resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.main.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06" # TLS 1.3 preferred
  certificate_arn   = var.acm_certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

variable "acm_certificate_arn" {
  description = "ACM certificate ARN for aidataboon.com (create via ACM console or Terraform aws_acm_certificate)"
  type        = string
  default     = "" # set in terraform.tfvars
}

# ---------------------------------------------------------------------------
# ECS Service + Auto-Scaling
# ---------------------------------------------------------------------------

resource "aws_ecs_service" "api" {
  name            = "${var.project_name}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.rag_api.arn

  # Fargate Spot (70%) + Fargate on-demand (30%) for reliability
  # Spot can save up to 70% on compute costs
  capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight            = 70
    base              = 0
  }

  capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 30
    base              = 1 # always keep at least 1 on-demand task
  }

  desired_count                      = 1
  health_check_grace_period_seconds  = 120

  network_configuration {
    subnets          = [aws_subnet.public_a.id, aws_subnet.public_b.id]
    security_groups  = [aws_security_group.ecs_sg.id]
    assign_public_ip = true # required when no NAT gateway
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "rag-api"
    container_port   = 8000
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true # auto-rollback on failed deployment
  }

  deployment_controller {
    type = "ECS" # use CODE_DEPLOY for blue/green with zero downtime
  }

  lifecycle {
    ignore_changes = [task_definition, desired_count] # managed by CI/CD
  }

  tags = { Name = "${var.project_name}-api-service" }
}

# Auto-scaling: min=1, max=4, scale out at 70% CPU
resource "aws_appautoscaling_target" "ecs_api" {
  max_capacity       = 4
  min_capacity       = 1
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "ecs_cpu" {
  name               = "${var.project_name}-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_api.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_api.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_api.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value       = 70.0 # scale out when avg CPU > 70%
    scale_in_cooldown  = 300  # 5 min cooldown prevents thrashing
    scale_out_cooldown = 60

    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
  }
}

# ---------------------------------------------------------------------------
# IAM: Least Privilege Roles
# ---------------------------------------------------------------------------

# ECS Task Role: what the application code can do
resource "aws_iam_role" "ecs_task" {
  name = "${var.project_name}-ecs-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Service = "ecs-tasks.amazonaws.com" }
        Action    = "sts:AssumeRole"
      }
    ]
  })

  tags = { Name = "${var.project_name}-ecs-task-role" }
}

# Least privilege: read/write only to project-owned S3 buckets
resource "aws_iam_role_policy" "ecs_task_s3" {
  name = "${var.project_name}-ecs-task-s3"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
        ]
        Resource = [
          aws_s3_bucket.data_lake.arn,
          "${aws_s3_bucket.data_lake.arn}/*",
          aws_s3_bucket.mlflow_artifacts.arn,
          "${aws_s3_bucket.mlflow_artifacts.arn}/*",
        ]
      }
    ]
  })
}

# ECS Execution Role: what ECS agent needs (pull images, push logs)
resource "aws_iam_role" "ecs_execution" {
  name = "${var.project_name}-ecs-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Service = "ecs-tasks.amazonaws.com" }
        Action    = "sts:AssumeRole"
      }
    ]
  })

  tags = { Name = "${var.project_name}-ecs-execution-role" }
}

resource "aws_iam_role_policy_attachment" "ecs_execution_managed" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Grant execution role access to SSM parameters (for secrets injection)
resource "aws_iam_role_policy" "ecs_execution_ssm" {
  name = "${var.project_name}-ecs-execution-ssm"
  role = aws_iam_role.ecs_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ssm:GetParameter", "ssm:GetParameters"]
        Resource = "arn:aws:ssm:${var.region}:${data.aws_caller_identity.current.account_id}:parameter/aidataboon/*"
      }
    ]
  })
}

# OIDC Provider for GitHub Actions; no static AWS credentials needed
# See: https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services
resource "aws_iam_openid_connect_provider" "github_actions" {
  url = "https://token.actions.githubusercontent.com"

  client_id_list = ["sts.amazonaws.com"]

  # GitHub Actions OIDC thumbprint (stable, rarely changes)
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]

  tags = { Name = "${var.project_name}-github-oidc" }
}

# GitHub Actions Deploy Role: assumed via OIDC (zero static credentials)
resource "aws_iam_role" "github_actions_deploy" {
  name = "${var.project_name}-github-actions-deploy"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = aws_iam_openid_connect_provider.github_actions.arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          }
          StringLike = {
            # Restrict to your specific repo; change to your GitHub org/repo
            "token.actions.githubusercontent.com:sub" = "repo:yourusername/aidataboon:*"
          }
        }
      }
    ]
  })

  tags = { Name = "${var.project_name}-github-deploy-role" }
}

resource "aws_iam_role_policy" "github_actions_deploy" {
  name = "${var.project_name}-github-deploy-policy"
  role = aws_iam_role.github_actions_deploy.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ECRAuth"
        Effect = "Allow"
        Action = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Sid    = "ECRPush"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:PutImage",
        ]
        Resource = aws_ecr_repository.api.arn
      },
      {
        Sid    = "ECSUpdateService"
        Effect = "Allow"
        Action = [
          "ecs:UpdateService",
          "ecs:DescribeServices",
          "ecs:RegisterTaskDefinition",
          "ecs:DescribeTaskDefinition",
          "ecs:ListTaskDefinitions",
        ]
        Resource = [
          aws_ecs_cluster.main.arn,
          aws_ecs_service.api.id,
          "arn:aws:ecs:${var.region}:${data.aws_caller_identity.current.account_id}:task-definition/${var.project_name}-*",
        ]
      },
      {
        Sid    = "PassRoleForTaskDef"
        Effect = "Allow"
        Action = "iam:PassRole"
        Resource = [
          aws_iam_role.ecs_task.arn,
          aws_iam_role.ecs_execution.arn,
        ]
      },
      {
        Sid    = "SSMUpdateParams"
        Effect = "Allow"
        Action = ["ssm:PutParameter", "ssm:GetParameter"]
        Resource = "arn:aws:ssm:${var.region}:${data.aws_caller_identity.current.account_id}:parameter/aidataboon/*"
      },
    ]
  })
}

# ---------------------------------------------------------------------------
# Secrets + Config: SSM Parameter Store (free for standard params)
# ---------------------------------------------------------------------------
# Alternative would be Secrets Manager at $0.40/secret/month.
# SSM Parameter Store standard params are FREE. Use SecureString for encryption.

resource "aws_ssm_parameter" "db_password" {
  name        = "/aidataboon/db_password"
  description = "RDS PostgreSQL master password for AIDataBoon"
  type        = "SecureString" # encrypted with AWS-managed KMS key (free)
  value       = var.db_password

  tags = { Name = "${var.project_name}-db-password" }

  lifecycle {
    ignore_changes = [value] # manage password rotation separately
  }
}

resource "aws_ssm_parameter" "redis_url" {
  name        = "/aidataboon/redis_url"
  description = "Redis connection URL (e.g. Upstash free tier)"
  type        = "SecureString"
  value       = "redis://placeholder:password@localhost:6379" # update after provisioning

  tags = { Name = "${var.project_name}-redis-url" }

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "airflow_fernet_key" {
  name        = "/aidataboon/airflow_fernet_key"
  description = "Airflow fernet key for encrypting connection passwords"
  type        = "SecureString"
  value       = "placeholder-generate-with-python-cryptography-fernet" # update before use

  tags = { Name = "${var.project_name}-airflow-fernet-key" }

  lifecycle {
    ignore_changes = [value]
  }
}

# ---------------------------------------------------------------------------
# Monitoring: CloudWatch (minimal cost configuration)
# ---------------------------------------------------------------------------

# 7-day retention reduces log storage cost significantly.
# Cost: $0.50/GB ingested + $0.03/GB stored/month
# At 7-day retention, stored cost is minimal for low-traffic services.
resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${var.project_name}"
  retention_in_days = 7 # Cost: 7 days keeps stored cost <$0.05/month for small workloads

  tags = { Name = "${var.project_name}-ecs-logs" }
}

# SNS Topic for alerts → email
resource "aws_sns_topic" "alerts" {
  name = "${var.project_name}-alerts"
  tags = { Name = "${var.project_name}-alerts" }
}

resource "aws_sns_topic_subscription" "email_alert" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = "ops@example.com" # replace with your ops email
}

# Alert: ALB 5XX error rate > 1% over 5 minutes
# Cost: $0.10/alarm/month (CloudWatch standard alarm)
resource "aws_cloudwatch_metric_alarm" "alb_5xx_rate" {
  alarm_name          = "${var.project_name}-alb-5xx-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  threshold           = 1.0 # 1% error rate

  metric_query {
    id          = "error_rate"
    expression  = "errors / requests * 100"
    label       = "ALB 5XX Error Rate (%)"
    return_data = true
  }

  metric_query {
    id = "errors"
    metric {
      namespace   = "AWS/ApplicationELB"
      metric_name = "HTTPCode_Target_5XX_Count"
      period      = 300
      stat        = "Sum"
      dimensions = {
        LoadBalancer = aws_lb.main.arn_suffix
      }
    }
  }

  metric_query {
    id = "requests"
    metric {
      namespace   = "AWS/ApplicationELB"
      metric_name = "RequestCount"
      period      = 300
      stat        = "Sum"
      dimensions = {
        LoadBalancer = aws_lb.main.arn_suffix
      }
    }
  }

  alarm_description = "ALB 5XX error rate exceeded 1%; check ECS service logs"
  alarm_actions     = [aws_sns_topic.alerts.arn]
  ok_actions        = [aws_sns_topic.alerts.arn]

  treat_missing_data = "notBreaching"

  tags = { Name = "${var.project_name}-alb-5xx-alarm" }
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "alb_dns_name" {
  description = "ALB DNS name: point aidataboon.com CNAME here (or use Route53 alias)"
  value       = aws_lb.main.dns_name
}

output "rds_endpoint" {
  description = "RDS PostgreSQL endpoint (host:port)"
  value       = "${aws_db_instance.postgres.address}:${aws_db_instance.postgres.port}"
  sensitive   = false # host is not sensitive; password is in SSM
}

output "ecs_cluster_name" {
  description = "ECS cluster name for use in CI/CD deployment commands"
  value       = aws_ecs_cluster.main.name
}

output "ecr_repository_url" {
  description = "ECR repository URL: use as docker push target in CI/CD"
  value       = aws_ecr_repository.api.repository_url
}

output "task_role_arn" {
  description = "ECS task IAM role ARN; attach additional policies here if needed"
  value       = aws_iam_role.ecs_task.arn
}

output "github_actions_role_arn" {
  description = "OIDC role ARN for GitHub Actions; set as AWS_DEPLOY_ROLE_ARN in repo vars"
  value       = aws_iam_role.github_actions_deploy.arn
}
