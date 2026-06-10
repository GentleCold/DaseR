# DaseR Monitoring Compose

This directory starts the metrics stack for externally running DaseR and vLLM
processes. It does not start DaseR or vLLM.

## Prerequisites

Start vLLM and DaseR separately. DaseR must expose Prometheus metrics at
`http://127.0.0.1:2026/metrics`.

## Start

From this directory:

```bash
cp .env.example .env
docker compose --env-file .env up -d
```

Open:

- Prometheus: <http://127.0.0.1:9090>
- Grafana: <http://127.0.0.1:3000>

The default Grafana login is `admin` / `admin`. Change
`GRAFANA_ADMIN_PASSWORD` in `.env` before starting the stack on a shared
machine.

## Runtime Data

By default, Prometheus and Grafana state is stored under:

```text
/data/zwt/daser_monitoring/
```

Override the location in `.env`:

```bash
DASER_MONITORING_DATA_ROOT=/data/zwt/daser_monitoring_dev
```

## Scrape Target

Prometheus scrapes DaseR through Docker's host gateway alias:

```text
host.docker.internal:2026
```

If DaseR runs on another host or port, edit
`prometheus/prometheus.yml` before starting the stack.
