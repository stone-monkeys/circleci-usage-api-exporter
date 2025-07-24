<h1 align="center">
  <br>
  <a href="https://www.circleci.com/"><img src="./circleci-logo.png" alt="CircleCI" width="200"></a>
  <br>
  Usage API Exporter
  <br>
</h1>

<h4 align="center">Open-source tools and examples for working with CircleCI's <a href="https://circleci.com/docs/api/v2/index.html#tag/Usage" target="_blank">Usage API</a> to optimize costs and improve pipeline performance.</h4>

<p align="center">
  <a href="#introduction">Introduction</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#whats-included">What's Included</a> •
  <a href="#minimum-requirements">Minimum Requirements</a> •
    <a href="#documentation">Documentation</a> •
  <a href="#contributing">Contributing</a>
</p>


> This repository is part of CircleCI Labs - solutions developed by CircleCI's field engineering team based on real customer needs.
> 
> ✅ **Created by Field Engineers @ CircleCI**  
> ✅ **Used by real CircleCI customers**  
> ❌ **NOT officially supported by CircleCI support**

## Introduction

CircleCI's Usage API provides powerful data about your CI/CD pipelines. This toolkit helps you quickly turn that data into actionable insights with ready-to-use scripts, analysis templates, and visualization integrations.

### What you can discover:

* Which jobs are burning through your budget 💰
* Where your pipelines are slowest 🐌
* Which resources are underutilized 📉
* How to right-size your compute classes ⚡

## Quick Start

```bash
git clone git@github.com:CircleCI-Labs/circleci-usage-api-exporter.git
cd circleci-usage-api-exporter

# These env vars are read by get_usage_report.py
export ORG_ID=""
export CIRCLECI_API_TOKEN=""
export START_DATE="2025-07-01"
export END_DATE="2025-06-01"

# Export your usage data
python scripts/get_usage_report.py
```

This will download a raw CSV dataset locally ready for analysis.

## What's Included

* **Utility to download usage data** - Quickly export data from the Usage API
* **Data processing scripts** - Clean and transform raw exports for analysis  
* **Visualization examples** - Templates for popular BI tools and custom dashboards

## Minimum Requirements

* Python 3.8+
* A CircleCI personal API token ([get yours here](https://app.circleci.com/settings/user/tokens))
* An organisation ID (find this in "Organization Settings")

## Documentation

* [**API Reference**](https://circleci.com/docs/api/v2/index.html#tag/Usage) - Usage API endpoints and data schema
* **[Examples](examples/)** - BI tool templates, analysis notebooks, and integration guides

## Contributing

### Ways to Contribute

* Request additions
* Add new visualization templates
* Improve analysis algorithms
* Share real-world optimization stories
* Fix bugs or improve documentation
* Add support for other BI tools