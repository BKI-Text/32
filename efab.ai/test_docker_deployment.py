#!/usr/bin/env python3
"""
Test Docker-based ML Model Deployment Pipeline
Beverly Knits AI Supply Chain Planner
"""

import os
import sys
import logging
import subprocess
import time
import requests
import json
from pathlib import Path
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DockerDeploymentTester:
    """Test Docker deployment pipeline"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.docker_dir = self.base_dir / "docker"
        self.test_results = []
        
    def run_command(self, command: str, timeout: int = 60) -> tuple:
        """Run shell command and return (success, output, error)"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.base_dir
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return False, "", str(e)
    
    def test_docker_availability(self) -> bool:
        """Test if Docker is available"""
        logger.info("🚀 Testing Docker availability")
        
        try:
            # Check Docker
            success, output, error = self.run_command("docker --version")
            if not success:
                logger.error(f"Docker not available: {error}")
                return False
            
            logger.info(f"Docker version: {output.strip()}")
            
            # Check Docker Compose
            success, output, error = self.run_command("docker-compose --version")
            if not success:
                logger.error(f"Docker Compose not available: {error}")
                return False
            
            logger.info(f"Docker Compose version: {output.strip()}")
            
            # Check if Docker daemon is running
            success, output, error = self.run_command("docker info")
            if not success:
                logger.error(f"Docker daemon not running: {error}")
                return False
            
            logger.info("✅ Docker and Docker Compose are available")
            return True
            
        except Exception as e:
            logger.error(f"❌ Docker availability test failed: {e}")
            return False
    
    def test_dockerfile_syntax(self) -> bool:
        """Test Dockerfile syntax"""
        logger.info("🚀 Testing Dockerfile syntax")
        
        try:
            dockerfile_path = self.docker_dir / "Dockerfile"
            if not dockerfile_path.exists():
                logger.error("Dockerfile not found")
                return False
            
            # Test Dockerfile syntax by attempting to build
            success, output, error = self.run_command(
                f"docker build -f {dockerfile_path} --target base -t beverly-knits-test:base .",
                timeout=300
            )
            
            if not success:
                logger.error(f"Dockerfile syntax error: {error}")
                return False
            
            logger.info("✅ Dockerfile syntax is valid")
            return True
            
        except Exception as e:
            logger.error(f"❌ Dockerfile syntax test failed: {e}")
            return False
    
    def test_docker_compose_syntax(self) -> bool:
        """Test Docker Compose file syntax"""
        logger.info("🚀 Testing Docker Compose syntax")
        
        try:
            compose_files = [
                "docker-compose.yml",
                "docker-compose.dev.yml",
                "docker-compose.prod.yml"
            ]
            
            for compose_file in compose_files:
                compose_path = self.docker_dir / compose_file
                if not compose_path.exists():
                    logger.warning(f"Compose file not found: {compose_file}")
                    continue
                
                # Test compose file syntax
                success, output, error = self.run_command(
                    f"docker-compose -f {compose_path} config",
                    timeout=30
                )
                
                if not success:
                    logger.error(f"Compose file syntax error in {compose_file}: {error}")
                    return False
                
                logger.info(f"✅ {compose_file} syntax is valid")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Docker Compose syntax test failed: {e}")
            return False
    
    def test_build_images(self) -> bool:
        """Test building Docker images"""
        logger.info("🚀 Testing Docker image builds")
        
        try:
            # Test building different stages
            stages = [
                ("base", "Base image build"),
                ("development", "Development image build"),
                ("production", "Production image build"),
                ("ml-training", "ML training image build"),
                ("ml-inference", "ML inference image build")
            ]
            
            for stage, description in stages:
                logger.info(f"Building {description}...")
                
                success, output, error = self.run_command(
                    f"docker build -f docker/Dockerfile --target {stage} -t beverly-knits-test:{stage} .",
                    timeout=600
                )
                
                if not success:
                    logger.error(f"Failed to build {stage} image: {error}")
                    return False
                
                logger.info(f"✅ {description} successful")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Docker image build test failed: {e}")
            return False
    
    def test_container_health(self) -> bool:
        """Test container health"""
        logger.info("🚀 Testing container health")
        
        try:
            # Start a test container
            container_name = "beverly-knits-health-test"
            
            # Clean up any existing container
            self.run_command(f"docker rm -f {container_name}")
            
            # Start container
            success, output, error = self.run_command(
                f"docker run -d --name {container_name} "
                f"-p 8080:8000 "
                f"-e ENVIRONMENT=test "
                f"beverly-knits-test:development",
                timeout=60
            )
            
            if not success:
                logger.error(f"Failed to start container: {error}")
                return False
            
            # Wait for container to be ready
            logger.info("Waiting for container to be ready...")
            time.sleep(10)
            
            # Check if container is running
            success, output, error = self.run_command(
                f"docker ps --filter name={container_name} --filter status=running --format '{{.Names}}'",
                timeout=30
            )
            
            if not success or container_name not in output:
                logger.error(f"Container not running: {error}")
                # Get container logs for debugging
                logs_success, logs_output, logs_error = self.run_command(
                    f"docker logs {container_name}",
                    timeout=30
                )
                if logs_success:
                    logger.error(f"Container logs: {logs_output}")
                return False
            
            # Test health endpoint
            try:
                response = requests.get("http://localhost:8080/health", timeout=10)
                if response.status_code != 200:
                    logger.error(f"Health check failed: {response.status_code}")
                    return False
                
                logger.info("✅ Container health check passed")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Health check request failed: {e}")
                return False
            
            finally:
                # Clean up container
                self.run_command(f"docker rm -f {container_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Container health test failed: {e}")
            return False
    
    def test_deployment_scripts(self) -> bool:
        """Test deployment scripts"""
        logger.info("🚀 Testing deployment scripts")
        
        try:
            scripts_dir = self.docker_dir / "scripts"
            scripts = [
                "deploy.sh",
                "train_models.sh"
            ]
            
            for script in scripts:
                script_path = scripts_dir / script
                if not script_path.exists():
                    logger.error(f"Script not found: {script}")
                    return False
                
                # Check if script is executable
                if not os.access(script_path, os.X_OK):
                    logger.error(f"Script not executable: {script}")
                    return False
                
                # Test script syntax (dry run)
                success, output, error = self.run_command(
                    f"bash -n {script_path}",
                    timeout=30
                )
                
                if not success:
                    logger.error(f"Script syntax error in {script}: {error}")
                    return False
                
                logger.info(f"✅ {script} syntax is valid")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment scripts test failed: {e}")
            return False
    
    def test_docker_compose_services(self) -> bool:
        """Test Docker Compose services"""
        logger.info("🚀 Testing Docker Compose services")
        
        try:
            # Use development compose file for testing
            compose_file = self.docker_dir / "docker-compose.dev.yml"
            
            # Validate compose file
            success, output, error = self.run_command(
                f"docker-compose -f {compose_file} config",
                timeout=30
            )
            
            if not success:
                logger.error(f"Compose file validation failed: {error}")
                return False
            
            # Parse compose configuration
            config = yaml.safe_load(output)
            services = config.get('services', {})
            
            expected_services = [
                'app-dev',
                'postgres-dev',
                'redis-dev'
            ]
            
            for service in expected_services:
                if service not in services:
                    logger.error(f"Expected service not found: {service}")
                    return False
                
                logger.info(f"✅ Service {service} is configured")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Docker Compose services test failed: {e}")
            return False
    
    def test_nginx_configuration(self) -> bool:
        """Test Nginx configuration"""
        logger.info("🚀 Testing Nginx configuration")
        
        try:
            nginx_conf = self.docker_dir / "nginx.conf"
            
            if not nginx_conf.exists():
                logger.error("Nginx configuration not found")
                return False
            
            # Test Nginx configuration syntax
            success, output, error = self.run_command(
                f"docker run --rm -v {nginx_conf}:/etc/nginx/nginx.conf nginx:alpine nginx -t",
                timeout=30
            )
            
            if not success:
                logger.error(f"Nginx configuration error: {error}")
                return False
            
            logger.info("✅ Nginx configuration is valid")
            return True
            
        except Exception as e:
            logger.error(f"❌ Nginx configuration test failed: {e}")
            return False
    
    def test_monitoring_configuration(self) -> bool:
        """Test monitoring configuration"""
        logger.info("🚀 Testing monitoring configuration")
        
        try:
            # Test Prometheus configuration
            prometheus_conf = self.docker_dir / "prometheus.yml"
            
            if prometheus_conf.exists():
                with open(prometheus_conf, 'r') as f:
                    config = yaml.safe_load(f)
                
                if 'scrape_configs' not in config:
                    logger.error("Prometheus scrape configs not found")
                    return False
                
                logger.info("✅ Prometheus configuration is valid")
            else:
                logger.warning("Prometheus configuration not found")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Monitoring configuration test failed: {e}")
            return False
    
    def test_environment_variables(self) -> bool:
        """Test environment variable configuration"""
        logger.info("🚀 Testing environment variables")
        
        try:
            # Check if .env files are properly referenced
            compose_files = [
                self.docker_dir / "docker-compose.yml",
                self.docker_dir / "docker-compose.dev.yml",
                self.docker_dir / "docker-compose.prod.yml"
            ]
            
            required_env_vars = [
                'ENVIRONMENT',
                'DB_HOST',
                'DB_NAME',
                'DB_USER'
            ]
            
            for compose_file in compose_files:
                if not compose_file.exists():
                    continue
                
                with open(compose_file, 'r') as f:
                    content = f.read()
                
                # Check for environment variable references
                for env_var in required_env_vars:
                    if env_var not in content:
                        logger.warning(f"Environment variable {env_var} not found in {compose_file.name}")
                
                logger.info(f"✅ Environment variables checked in {compose_file.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment variables test failed: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all deployment tests"""
        logger.info("🚀 Running Docker Deployment Pipeline Tests")
        
        tests = [
            ("Docker Availability", self.test_docker_availability),
            ("Dockerfile Syntax", self.test_dockerfile_syntax),
            ("Docker Compose Syntax", self.test_docker_compose_syntax),
            ("Build Images", self.test_build_images),
            ("Container Health", self.test_container_health),
            ("Deployment Scripts", self.test_deployment_scripts),
            ("Docker Compose Services", self.test_docker_compose_services),
            ("Nginx Configuration", self.test_nginx_configuration),
            ("Monitoring Configuration", self.test_monitoring_configuration),
            ("Environment Variables", self.test_environment_variables)
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
        logger.info(f"DOCKER DEPLOYMENT TESTS SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Passed: {passed}/{total}")
        logger.info(f"Success rate: {passed/total*100:.1f}%")
        
        for test_name, result in self.test_results:
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"  {test_name}: {status}")
        
        if passed == total:
            logger.info("🎉 All Docker deployment tests passed!")
            return True
        else:
            logger.error("💥 Some Docker deployment tests failed!")
            return False

def main():
    """Main test execution"""
    tester = DockerDeploymentTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()