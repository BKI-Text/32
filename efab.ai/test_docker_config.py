#!/usr/bin/env python3
"""
Test Docker Configuration Files (without requiring Docker)
Beverly Knits AI Supply Chain Planner
"""

import os
import sys
import logging
import yaml
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DockerConfigTester:
    """Test Docker configuration files without Docker"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.docker_dir = self.base_dir / "docker"
        self.test_results = []
        
    def test_dockerfile_structure(self) -> bool:
        """Test Dockerfile structure"""
        logger.info("🚀 Testing Dockerfile structure")
        
        try:
            dockerfile_path = self.docker_dir / "Dockerfile"
            
            if not dockerfile_path.exists():
                logger.error("Dockerfile not found")
                return False
            
            with open(dockerfile_path, 'r') as f:
                content = f.read()
            
            # Check for required stages
            required_stages = [
                "FROM python:3.11-slim as base",
                "FROM base as development", 
                "FROM base as production",
                "FROM base as ml-training",
                "FROM base as ml-inference"
            ]
            
            for stage in required_stages:
                if stage not in content:
                    logger.error(f"Required stage not found: {stage}")
                    return False
            
            # Check for security practices
            security_checks = [
                "RUN groupadd -r",  # Non-root user creation
                "USER ",             # User switching
                "HEALTHCHECK"        # Health check
            ]
            
            for check in security_checks:
                if check not in content:
                    logger.warning(f"Security practice not found: {check}")
            
            logger.info("✅ Dockerfile structure is valid")
            return True
            
        except Exception as e:
            logger.error(f"❌ Dockerfile structure test failed: {e}")
            return False
    
    def test_compose_files(self) -> bool:
        """Test Docker Compose files structure"""
        logger.info("🚀 Testing Docker Compose files")
        
        try:
            compose_files = [
                ("docker-compose.yml", "Main"),
                ("docker-compose.dev.yml", "Development"),
                ("docker-compose.prod.yml", "Production")
            ]
            
            for compose_file, description in compose_files:
                compose_path = self.docker_dir / compose_file
                
                if not compose_path.exists():
                    logger.error(f"{description} compose file not found: {compose_file}")
                    return False
                
                with open(compose_path, 'r') as f:
                    try:
                        config = yaml.safe_load(f)
                    except yaml.YAMLError as e:
                        logger.error(f"YAML syntax error in {compose_file}: {e}")
                        return False
                
                # Check basic structure
                if 'version' not in config:
                    logger.error(f"No version specified in {compose_file}")
                    return False
                
                if 'services' not in config:
                    logger.error(f"No services defined in {compose_file}")
                    return False
                
                # Check for required services based on file type
                if "prod" in compose_file:
                    required_services = ['app', 'postgres', 'redis', 'nginx']
                elif "dev" in compose_file:
                    required_services = ['app-dev', 'postgres-dev', 'redis-dev']
                else:
                    required_services = ['app', 'postgres', 'redis']
                
                services = config.get('services', {})
                for service in required_services:
                    if service not in services:
                        logger.warning(f"Expected service not found in {compose_file}: {service}")
                
                logger.info(f"✅ {description} compose file is valid")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Docker Compose files test failed: {e}")
            return False
    
    def test_nginx_config(self) -> bool:
        """Test Nginx configuration"""
        logger.info("🚀 Testing Nginx configuration")
        
        try:
            nginx_conf = self.docker_dir / "nginx.conf"
            
            if not nginx_conf.exists():
                logger.error("Nginx configuration not found")
                return False
            
            with open(nginx_conf, 'r') as f:
                content = f.read()
            
            # Check for required directives
            required_directives = [
                "upstream app_backend",
                "upstream ml_backend",
                "server {",
                "location / {",
                "location /api/v1/ml/",
                "location /api/v1/ws/",
                "proxy_pass"
            ]
            
            for directive in required_directives:
                if directive not in content:
                    logger.error(f"Required directive not found: {directive}")
                    return False
            
            # Check for security headers
            security_headers = [
                "X-Frame-Options",
                "X-Content-Type-Options",
                "X-XSS-Protection",
                "Strict-Transport-Security"
            ]
            
            for header in security_headers:
                if header not in content:
                    logger.warning(f"Security header not found: {header}")
            
            logger.info("✅ Nginx configuration is valid")
            return True
            
        except Exception as e:
            logger.error(f"❌ Nginx configuration test failed: {e}")
            return False
    
    def test_prometheus_config(self) -> bool:
        """Test Prometheus configuration"""
        logger.info("🚀 Testing Prometheus configuration")
        
        try:
            prometheus_conf = self.docker_dir / "prometheus.yml"
            
            if not prometheus_conf.exists():
                logger.error("Prometheus configuration not found")
                return False
            
            with open(prometheus_conf, 'r') as f:
                try:
                    config = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    logger.error(f"YAML syntax error in prometheus.yml: {e}")
                    return False
            
            # Check required sections
            required_sections = ['global', 'scrape_configs']
            for section in required_sections:
                if section not in config:
                    logger.error(f"Required section not found: {section}")
                    return False
            
            # Check scrape configs
            scrape_configs = config.get('scrape_configs', [])
            if not scrape_configs:
                logger.error("No scrape configs found")
                return False
            
            # Check for expected jobs
            expected_jobs = [
                'beverly-knits-app',
                'beverly-knits-ml',
                'postgres',
                'redis'
            ]
            
            job_names = [job.get('job_name') for job in scrape_configs]
            for job in expected_jobs:
                if job not in job_names:
                    logger.warning(f"Expected job not found: {job}")
            
            logger.info("✅ Prometheus configuration is valid")
            return True
            
        except Exception as e:
            logger.error(f"❌ Prometheus configuration test failed: {e}")
            return False
    
    def test_deployment_scripts(self) -> bool:
        """Test deployment scripts"""
        logger.info("🚀 Testing deployment scripts")
        
        try:
            scripts_dir = self.docker_dir / "scripts"
            
            if not scripts_dir.exists():
                logger.error("Scripts directory not found")
                return False
            
            scripts = [
                ("deploy.sh", "Main deployment script"),
                ("train_models.sh", "Model training script")
            ]
            
            for script_name, description in scripts:
                script_path = scripts_dir / script_name
                
                if not script_path.exists():
                    logger.error(f"{description} not found: {script_name}")
                    return False
                
                # Check if script is executable
                if not os.access(script_path, os.X_OK):
                    logger.warning(f"Script not executable: {script_name}")
                
                with open(script_path, 'r') as f:
                    content = f.read()
                
                # Check for bash shebang
                if not content.startswith('#!/bin/bash'):
                    logger.warning(f"Script missing bash shebang: {script_name}")
                
                # Check for basic functions
                if script_name == "deploy.sh":
                    required_functions = [
                        "check_docker()",
                        "start_services()",
                        "stop_services()",
                        "health_check()"
                    ]
                elif script_name == "train_models.sh":
                    required_functions = [
                        "train_models()",
                        "evaluate_models()",
                        "deploy_models()"
                    ]
                
                for func in required_functions:
                    if func not in content:
                        logger.warning(f"Function not found in {script_name}: {func}")
                
                logger.info(f"✅ {description} is valid")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment scripts test failed: {e}")
            return False
    
    def test_documentation(self) -> bool:
        """Test documentation"""
        logger.info("🚀 Testing documentation")
        
        try:
            readme_path = self.docker_dir / "README.md"
            
            if not readme_path.exists():
                logger.error("Docker README.md not found")
                return False
            
            with open(readme_path, 'r') as f:
                content = f.read()
            
            # Check for required sections
            required_sections = [
                "# Docker-based ML Model Deployment Pipeline",
                "## Quick Start",
                "## Configuration",
                "## Deployment Commands",
                "## Monitoring and Observability",
                "## Security",
                "## Troubleshooting"
            ]
            
            for section in required_sections:
                if section not in content:
                    logger.warning(f"Documentation section not found: {section}")
            
            logger.info("✅ Documentation is comprehensive")
            return True
            
        except Exception as e:
            logger.error(f"❌ Documentation test failed: {e}")
            return False
    
    def test_directory_structure(self) -> bool:
        """Test directory structure"""
        logger.info("🚀 Testing directory structure")
        
        try:
            # Check for required files
            required_files = [
                "Dockerfile",
                "docker-compose.yml",
                "docker-compose.dev.yml",
                "docker-compose.prod.yml",
                "nginx.conf",
                "prometheus.yml",
                "README.md"
            ]
            
            for file_name in required_files:
                file_path = self.docker_dir / file_name
                if not file_path.exists():
                    logger.error(f"Required file not found: {file_name}")
                    return False
            
            # Check for scripts directory
            scripts_dir = self.docker_dir / "scripts"
            if not scripts_dir.exists():
                logger.error("Scripts directory not found")
                return False
            
            # Check for required scripts
            required_scripts = ["deploy.sh", "train_models.sh"]
            for script in required_scripts:
                script_path = scripts_dir / script
                if not script_path.exists():
                    logger.error(f"Required script not found: {script}")
                    return False
            
            logger.info("✅ Directory structure is correct")
            return True
            
        except Exception as e:
            logger.error(f"❌ Directory structure test failed: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all configuration tests"""
        logger.info("🚀 Running Docker Configuration Tests")
        
        tests = [
            ("Directory Structure", self.test_directory_structure),
            ("Dockerfile Structure", self.test_dockerfile_structure),
            ("Docker Compose Files", self.test_compose_files),
            ("Nginx Configuration", self.test_nginx_config),
            ("Prometheus Configuration", self.test_prometheus_config),
            ("Deployment Scripts", self.test_deployment_scripts),
            ("Documentation", self.test_documentation)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Running test: {test_name}")
                logger.info(f"{'='*60}")
                
                result = test_func()
                self.test_results.append((test_name, result))
                
                if result:
                    passed += 1
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
                    
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")
                self.test_results.append((test_name, False))
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"DOCKER CONFIGURATION TESTS SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Passed: {passed}/{total}")
        logger.info(f"Success rate: {passed/total*100:.1f}%")
        
        for test_name, result in self.test_results:
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"  {test_name}: {status}")
        
        if passed == total:
            logger.info("🎉 All Docker configuration tests passed!")
            return True
        else:
            logger.error("💥 Some Docker configuration tests failed!")
            return False

def main():
    """Main test execution"""
    tester = DockerConfigTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()