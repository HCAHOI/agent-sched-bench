#!/usr/bin/env bash
set -euo pipefail

print_banner() {
  cat <<'EOF'
=======================================
  Mihomo 订阅代理配置助手
=======================================
EOF
}

print_disclaimer() {
  cat <<'EOF'
==============================================
              免责声明
==============================================
本脚本按原样提供，使用者需自行承担风险。
代理、订阅、Docker 与系统服务配置均会被修改，请先确认目标机器可接受这些变更。
==============================================
EOF
}

print_menu() {
  cat <<'EOF'

请选择要执行的命令：
------------------------------
  1. install   - 安装并配置 mihomo 订阅代理
  2. status    - 查看当前安装与服务状态
  3. check     - 检查服务、控制器、代理文件与 Docker 代理配置
  4. uninstall - 卸载 mihomo 与相关配置
  5. exit      - 退出
EOF
}

usage() {
  cat <<'EOF'
Install and configure mihomo from a Clash-compatible subscription URL.

Usage:
  sudo bash scripts/setup/install_mihomo_subscription.sh
  sudo bash scripts/setup/install_mihomo_subscription.sh install \
    --subscription-url 'https://example.com/api/v1/client/subscribe?...' \
    [--archive-path /path/to/mihomo-archive.gz] \
    [--version v1.19.23] \
    [--mixed-port 7890] \
    [--controller-port 9090] \
    [--config-dir /etc/mihomo] \
    [--binary-path /usr/local/bin/mihomo] \
    [--service-name mihomo] \
    [--configure-shell] \
    [--configure-docker] \
    [--dry-run]

  sudo bash scripts/setup/install_mihomo_subscription.sh status
  sudo bash scripts/setup/install_mihomo_subscription.sh check
  sudo bash scripts/setup/install_mihomo_subscription.sh uninstall \
    [--config-dir /etc/mihomo] \
    [--binary-path /usr/local/bin/mihomo] \
    [--service-name mihomo] \
    [--remove-shell-config] \
    [--remove-docker-config] \
    [--dry-run]

Legacy one-shot usage:
  sudo bash scripts/setup/install_mihomo_subscription.sh \
    --subscription-url 'https://example.com/api/v1/client/subscribe?...' \
    [--archive-path /path/to/mihomo-archive.gz] \
    [--version v1.19.23] \
    [--mixed-port 7890] \
    [--controller-port 9090] \
    [--config-dir /etc/mihomo] \
    [--binary-path /usr/local/bin/mihomo] \
    [--service-name mihomo] \
    [--configure-shell] \
    [--configure-docker] \
    [--dry-run]

Options:
  --subscription-url URL    Required unless MIHOMO_SUBSCRIPTION_URL is set.
  --archive-path PATH       Optional local archive path. Skip remote download when set.
  --version TAG             Optional release tag. Defaults to latest stable.
  --mixed-port PORT         Mihomo mixed-port. Default: 7890
  --controller-port PORT    External controller port. Default: 9090
  --config-dir PATH         Mihomo config directory. Default: /etc/mihomo
  --binary-path PATH        Mihomo binary path. Default: /usr/local/bin/mihomo
  --service-name NAME       systemd service name. Default: mihomo
  --configure-shell         Write /etc/profile.d/mihomo-proxy.sh
  --configure-docker        Write /etc/systemd/system/docker.service.d/http-proxy.conf
  --remove-shell-config     Remove /etc/profile.d/mihomo-proxy.sh on uninstall
  --remove-docker-config    Remove Docker proxy drop-in on uninstall
  --dry-run                 Print planned actions without writing files.
  -h, --help                Show this help.

Environment:
  MIHOMO_SUBSCRIPTION_URL   Alternative way to pass the subscription URL.
EOF
}

DEFAULT_VERSION="v1.19.23"

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "This script must run as root." >&2
    exit 1
  fi
}

need_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 1
  fi
}

prompt_yes_no() {
  local prompt="$1"
  local answer
  while true; do
    read -r -p "${prompt} [Yes/No] : " answer
    case "${answer}" in
      Y|y|Yes|YES|yes)
        return 0
        ;;
      N|n|No|NO|no)
        return 1
        ;;
      *)
        echo "请输入 Yes 或 No。"
        ;;
    esac
  done
}

prompt_menu_choice() {
  local answer
  while true; do
    read -r -p "请选择 (输入数字): " answer
    case "${answer}" in
      1) echo "install"; return 0 ;;
      2) echo "status"; return 0 ;;
      3) echo "check"; return 0 ;;
      4) echo "uninstall"; return 0 ;;
      5) echo "exit"; return 0 ;;
      *) echo "输入错误，请重新输入" ;;
    esac
  done
}

show_status() {
  local config_dir="${1:-/etc/mihomo}"
  local binary_path="${2:-/usr/local/bin/mihomo}"
  local service_name="${3:-mihomo}"

  print_banner
  echo "系统: $(uname -s) $(uname -m)"
  echo "二进制: ${binary_path}"
  echo "配置目录: ${config_dir}"
  echo

  if command -v "${binary_path}" >/dev/null 2>&1; then
    "${binary_path}" -v || true
  else
    echo "mihomo 二进制不存在"
  fi
  echo "---"

  if command -v systemctl >/dev/null 2>&1; then
    systemctl --no-pager --full status "${service_name}" || true
  else
    echo "systemctl 不可用"
  fi
  echo "---"

  if [[ -f "${config_dir}/config.yaml" ]]; then
    sed -n '1,80p' "${config_dir}/config.yaml"
  else
    echo "配置文件不存在: ${config_dir}/config.yaml"
  fi
  echo "---"

  if [[ -f "/etc/profile.d/${service_name}-proxy.sh" ]]; then
    echo "shell 代理配置存在: /etc/profile.d/${service_name}-proxy.sh"
  fi
  if [[ -f "/etc/systemd/system/docker.service.d/http-proxy.conf" ]]; then
    echo "docker 代理配置存在: /etc/systemd/system/docker.service.d/http-proxy.conf"
  fi
}

run_check() {
  local config_dir="${1:-/etc/mihomo}"
  local binary_path="${2:-/usr/local/bin/mihomo}"
  local service_name="${3:-mihomo}"
  local config_file="${config_dir}/config.yaml"
  local controller_addr="" mixed_port="" secret=""

  print_banner
  echo "执行检查..."
  echo "系统: $(uname -s) $(uname -m)"
  echo "---"

  if [[ -x "${binary_path}" ]]; then
    echo "✅ mihomo 二进制存在: ${binary_path}"
    "${binary_path}" -v || true
  else
    echo "❌ mihomo 二进制不存在: ${binary_path}"
  fi
  echo "---"

  if command -v systemctl >/dev/null 2>&1; then
    if systemctl is-active --quiet "${service_name}"; then
      echo "✅ 服务运行中: ${service_name}"
    else
      echo "❌ 服务未运行: ${service_name}"
    fi
  else
    echo "⚠️  systemctl 不可用，跳过服务检查"
  fi
  echo "---"

  if [[ -f "${config_file}" ]]; then
    echo "✅ 配置文件存在: ${config_file}"
    mixed_port="$(awk -F': ' '/^mixed-port:/{print $2; exit}' "${config_file}" | tr -d '\r')"
    controller_addr="$(awk -F': ' '/^external-controller:/{print $2; exit}' "${config_file}" | tr -d '\r')"
    secret="$(awk -F': ' '/^secret:/{print $2; exit}' "${config_file}" | tr -d '\r')"
    echo "mixed-port: ${mixed_port:-unknown}"
    echo "external-controller: ${controller_addr:-unknown}"
  else
    echo "❌ 配置文件不存在: ${config_file}"
  fi
  echo "---"

  if [[ -n "${controller_addr}" ]] && [[ -n "${secret}" ]] && command -v curl >/dev/null 2>&1; then
    if curl -fsS --max-time 5 \
      -H "Authorization: Bearer ${secret}" \
      "http://${controller_addr}/version" >/tmp/mihomo-check.$$ 2>/dev/null; then
      echo "✅ 控制器 API 可访问: http://${controller_addr}/version"
      sed -n '1,5p' "/tmp/mihomo-check.$$"
    else
      echo "❌ 控制器 API 不可访问: http://${controller_addr}/version"
    fi
    rm -f "/tmp/mihomo-check.$$"
  fi
  echo "---"

  if [[ -f "${config_dir}/proxy_providers/subscription.yaml" ]]; then
    echo "✅ 订阅缓存存在: ${config_dir}/proxy_providers/subscription.yaml"
  else
    echo "ℹ️  订阅缓存尚不存在: ${config_dir}/proxy_providers/subscription.yaml"
  fi

  if [[ -f "/etc/profile.d/${service_name}-proxy.sh" ]]; then
    echo "✅ shell 代理配置存在: /etc/profile.d/${service_name}-proxy.sh"
  else
    echo "ℹ️  shell 代理配置不存在"
  fi

  if [[ -f "/etc/systemd/system/docker.service.d/http-proxy.conf" ]]; then
    echo "✅ Docker 代理 drop-in 存在"
    if command -v systemctl >/dev/null 2>&1; then
      systemctl show docker --property=Environment || true
    fi
  else
    echo "ℹ️  Docker 代理 drop-in 不存在"
  fi
}

uninstall_mihomo() {
  local config_dir="$1"
  local binary_path="$2"
  local service_name="$3"
  local remove_shell="$4"
  local remove_docker="$5"

  if [[ "${DRY_RUN}" -eq 0 ]]; then
    require_root
  fi
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    need_cmd systemctl
  fi

  print_banner
  echo "准备卸载 ${service_name}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] systemctl disable --now ${service_name}"
    echo "[dry-run] rm -rf ${config_dir}"
    echo "[dry-run] rm -f ${binary_path}"
    if [[ "${remove_shell}" -eq 1 ]]; then
      echo "[dry-run] rm -f /etc/profile.d/${service_name}-proxy.sh"
    fi
    if [[ "${remove_docker}" -eq 1 ]]; then
      echo "[dry-run] rm -f /etc/systemd/system/docker.service.d/http-proxy.conf"
      echo "[dry-run] systemctl daemon-reload"
      echo "[dry-run] systemctl restart docker"
    fi
    return 0
  fi

  systemctl disable --now "${service_name}" || true
  rm -rf "${config_dir}"
  rm -f "${binary_path}"

  if [[ "${remove_shell}" -eq 1 ]]; then
    rm -f "/etc/profile.d/${service_name}-proxy.sh"
  fi
  if [[ "${remove_docker}" -eq 1 ]]; then
    rm -f "/etc/systemd/system/docker.service.d/http-proxy.conf"
    systemctl daemon-reload
    systemctl restart docker
  fi

  echo "卸载完成。"
}

detect_arch_candidates() {
  local arch
  arch="$(uname -m)"
  case "${arch}" in
    x86_64|amd64)
      printf '%s\n' "linux-amd64-compatible" "linux-amd64"
      ;;
    aarch64|arm64)
      printf '%s\n' "linux-arm64-v8" "linux-arm64"
      ;;
    armv7l|armv7)
      printf '%s\n' "linux-armv7"
      ;;
    i386|i686)
      printf '%s\n' "linux-386"
      ;;
    *)
      echo "Unsupported architecture: ${arch}" >&2
      exit 1
      ;;
  esac
}

resolve_latest_version() {
  local location
  location="$(
    curl -fsSIL -o /dev/null -w '%{url_effective}' \
      'https://github.com/MetaCubeX/mihomo/releases/latest'
  )"
  python3 -c '
import re
import sys
match = re.search(r"/releases/tag/([^/]+)$", sys.argv[1].strip())
if not match:
    raise SystemExit(f"Could not parse version tag from {sys.argv[1]!r}")
print(match.group(1))
' "${location}"
}

resolve_asset_url() {
  local version="$1"
  shift
  local suffix
  for suffix in "$@"; do
    local candidate
    candidate="https://github.com/MetaCubeX/mihomo/releases/download/${version}/mihomo-${suffix}-${version}.gz"
    if curl -fsSIL -o /dev/null "${candidate}" >/dev/null 2>&1; then
      echo "${candidate}"
      return 0
    fi
  done
  echo "Could not find a mihomo asset for version ${version} and suffixes: $*" >&2
  exit 1
}

run_cmd() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run] %q' "$1"
    shift
    for arg in "$@"; do
      printf ' %q' "${arg}"
    done
    printf '\n'
    return 0
  fi
  "$@"
}

write_file() {
  local path="$1"
  local mode="$2"
  shift 2
  local content="$*"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run] write %s (%s)\n' "${path}" "${mode}"
    return 0
  fi
  install -d "$(dirname "${path}")"
  umask 077
  cat > "${path}" <<EOF
${content}
EOF
  chmod "${mode}" "${path}"
}

main() {
  local mode=""
  SUBSCRIPTION_URL="${MIHOMO_SUBSCRIPTION_URL:-}"
  ARCHIVE_PATH="${MIHOMO_ARCHIVE_PATH:-}"
  VERSION=""
  MIXED_PORT="7890"
  CONTROLLER_PORT="9090"
  CONFIG_DIR="/etc/mihomo"
  BINARY_PATH="/usr/local/bin/mihomo"
  SERVICE_NAME="mihomo"
  CONFIGURE_SHELL=0
  CONFIGURE_DOCKER=0
  REMOVE_SHELL_CONFIG=0
  REMOVE_DOCKER_CONFIG=0
  DRY_RUN=0

  if [[ $# -gt 0 ]]; then
    case "$1" in
      install|status|check|uninstall)
        mode="$1"
        shift
        ;;
    esac
  fi

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --subscription-url)
        SUBSCRIPTION_URL="${2:-}"
        shift 2
        ;;
      --version)
        VERSION="${2:-}"
        shift 2
        ;;
      --archive-path)
        ARCHIVE_PATH="${2:-}"
        shift 2
        ;;
      --mixed-port)
        MIXED_PORT="${2:-}"
        shift 2
        ;;
      --controller-port)
        CONTROLLER_PORT="${2:-}"
        shift 2
        ;;
      --config-dir)
        CONFIG_DIR="${2:-}"
        shift 2
        ;;
      --binary-path)
        BINARY_PATH="${2:-}"
        shift 2
        ;;
      --service-name)
        SERVICE_NAME="${2:-}"
        shift 2
        ;;
      --configure-shell)
        CONFIGURE_SHELL=1
        shift
        ;;
      --configure-docker)
        CONFIGURE_DOCKER=1
        shift
        ;;
      --remove-shell-config)
        REMOVE_SHELL_CONFIG=1
        shift
        ;;
      --remove-docker-config)
        REMOVE_DOCKER_CONFIG=1
        shift
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "Unknown option: $1" >&2
        usage >&2
        exit 1
        ;;
    esac
  done

  if [[ -z "${mode}" ]]; then
    if [[ -t 0 && -z "${MIHOMO_NON_INTERACTIVE:-}" ]]; then
      print_banner
      print_disclaimer
      if ! prompt_yes_no "是否同意以上免责声明并继续使用？"; then
        echo "您已拒绝免责声明。"
        exit 1
      fi
      print_menu
      mode="$(prompt_menu_choice)"
      if [[ "${mode}" == "exit" ]]; then
        echo "已退出。"
        exit 0
      fi
    else
      mode="install"
    fi
  fi

  if [[ "${mode}" == "status" ]]; then
    show_status "${CONFIG_DIR}" "${BINARY_PATH}" "${SERVICE_NAME}"
    exit 0
  fi

  if [[ "${mode}" == "check" ]]; then
    run_check "${CONFIG_DIR}" "${BINARY_PATH}" "${SERVICE_NAME}"
    exit 0
  fi

  if [[ "${mode}" == "uninstall" ]]; then
    if [[ -t 0 && -z "${MIHOMO_NON_INTERACTIVE:-}" ]]; then
      if ! prompt_yes_no "确认要卸载 ${SERVICE_NAME} 吗？"; then
        echo "已取消。"
        exit 0
      fi
      if [[ "${REMOVE_SHELL_CONFIG}" -eq 0 ]] && prompt_yes_no "是否同时移除 shell 代理配置？"; then
        REMOVE_SHELL_CONFIG=1
      fi
      if [[ "${REMOVE_DOCKER_CONFIG}" -eq 0 ]] && prompt_yes_no "是否同时移除 Docker 代理配置？"; then
        REMOVE_DOCKER_CONFIG=1
      fi
    fi
    uninstall_mihomo \
      "${CONFIG_DIR}" \
      "${BINARY_PATH}" \
      "${SERVICE_NAME}" \
      "${REMOVE_SHELL_CONFIG}" \
      "${REMOVE_DOCKER_CONFIG}"
    exit 0
  fi

  print_banner
  echo "检测到系统: $(uname -s) $(uname -m)"

  if [[ -t 0 && -z "${MIHOMO_NON_INTERACTIVE:-}" ]]; then
    if [[ -z "${SUBSCRIPTION_URL}" ]]; then
      read -r -p "请输入 Clash/Mihomo 订阅 URL: " SUBSCRIPTION_URL
    fi
    if [[ "${CONFIGURE_SHELL}" -eq 0 ]] && prompt_yes_no "是否配置 shell 代理环境变量？"; then
      CONFIGURE_SHELL=1
    fi
    if [[ "${CONFIGURE_DOCKER}" -eq 0 ]] && prompt_yes_no "是否配置 Docker daemon 代理？"; then
      CONFIGURE_DOCKER=1
    fi
  fi

  if [[ "${DRY_RUN}" -eq 0 ]]; then
    require_root
  fi

  need_cmd curl
  need_cmd python3
  need_cmd gzip
  need_cmd install
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    need_cmd systemctl
  fi

  if [[ -z "${SUBSCRIPTION_URL}" ]]; then
    echo "A subscription URL is required." >&2
    exit 1
  fi

  local -a suffixes=()
  local version asset_url archive_path binary_tmp secret
  while IFS= read -r suffix; do
    suffixes+=("${suffix}")
  done < <(detect_arch_candidates)
  tmpdir="$(mktemp -d)"
  if [[ -n "${ARCHIVE_PATH}" ]]; then
    archive_path="${ARCHIVE_PATH}"
  else
    archive_path="${tmpdir}/mihomo.gz"
  fi
  binary_tmp="${tmpdir}/mihomo"
  secret="$(python3 - <<'PY'
import secrets
print(secrets.token_hex(16))
PY
)"

  trap 'rm -rf "'"${tmpdir}"'"' EXIT

  echo "Architecture candidates: ${suffixes[*]}"
  if [[ -n "${ARCHIVE_PATH}" ]]; then
    echo "Using local archive: ${archive_path}"
  else
    if [[ -n "${VERSION}" ]]; then
      version="${VERSION}"
    else
      version="$(resolve_latest_version || true)"
      if [[ -z "${version}" ]]; then
        version="${DEFAULT_VERSION}"
      fi
    fi
    asset_url="$(resolve_asset_url "${version}" "${suffixes[@]}")"
    echo "Using mihomo version: ${version}"
    echo "Downloading: ${asset_url}"
    run_cmd curl -fsSL "${asset_url}" -o "${archive_path}"
  fi

  if [[ "${DRY_RUN}" -eq 0 ]]; then
    gzip -dc "${archive_path}" > "${binary_tmp}"
    install -m 0755 "${binary_tmp}" "${BINARY_PATH}"
    install -d -m 0755 "${CONFIG_DIR}/proxy_providers"
  fi

  write_file "${CONFIG_DIR}/config.yaml" 0644 "$(cat <<EOF
mixed-port: ${MIXED_PORT}
allow-lan: false
mode: rule
log-level: info
ipv6: true
external-controller: 127.0.0.1:${CONTROLLER_PORT}
secret: ${secret}
profile:
  store-selected: true
  store-fake-ip: false

proxy-providers:
  subscription:
    type: http
    url: ${SUBSCRIPTION_URL}
    path: ./proxy_providers/subscription.yaml
    interval: 86400
    health-check:
      enable: true
      url: https://www.gstatic.com/generate_204
      interval: 600
      timeout: 5000

proxy-groups:
  - name: PROXY
    type: select
    use:
      - subscription
  - name: AUTO
    type: url-test
    use:
      - subscription
    url: https://www.gstatic.com/generate_204
    interval: 300

rules:
  - MATCH,PROXY
EOF
)"

  write_file "/etc/systemd/system/${SERVICE_NAME}.service" 0644 "$(cat <<EOF
[Unit]
Description=mihomo Daemon, Another Clash Kernel
After=network.target NetworkManager.service systemd-networkd.service iwd.service

[Service]
Type=simple
LimitNPROC=500
LimitNOFILE=1000000
CapabilityBoundingSet=CAP_NET_ADMIN CAP_NET_RAW CAP_NET_BIND_SERVICE CAP_SYS_TIME CAP_SYS_PTRACE CAP_DAC_READ_SEARCH CAP_DAC_OVERRIDE
AmbientCapabilities=CAP_NET_ADMIN CAP_NET_RAW CAP_NET_BIND_SERVICE CAP_SYS_TIME CAP_SYS_PTRACE CAP_DAC_READ_SEARCH CAP_DAC_OVERRIDE
Restart=always
ExecStartPre=/usr/bin/sleep 1s
ExecStart=${BINARY_PATH} -d ${CONFIG_DIR}
ExecReload=/bin/kill -HUP \$MAINPID

[Install]
WantedBy=multi-user.target
EOF
)"

  if [[ "${CONFIGURE_SHELL}" -eq 1 ]]; then
    write_file "/etc/profile.d/${SERVICE_NAME}-proxy.sh" 0644 "$(cat <<EOF
export HTTP_PROXY=http://127.0.0.1:${MIXED_PORT}
export HTTPS_PROXY=http://127.0.0.1:${MIXED_PORT}
export ALL_PROXY=socks5://127.0.0.1:${MIXED_PORT}
export http_proxy=\${HTTP_PROXY}
export https_proxy=\${HTTPS_PROXY}
export all_proxy=\${ALL_PROXY}
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy=\${NO_PROXY}
EOF
)"
  fi

  if [[ "${CONFIGURE_DOCKER}" -eq 1 ]]; then
    write_file "/etc/systemd/system/docker.service.d/http-proxy.conf" 0644 "$(cat <<EOF
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:${MIXED_PORT}"
Environment="HTTPS_PROXY=http://127.0.0.1:${MIXED_PORT}"
Environment="ALL_PROXY=socks5://127.0.0.1:${MIXED_PORT}"
Environment="NO_PROXY=localhost,127.0.0.1,::1"
EOF
)"
  fi

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] systemctl daemon-reload"
    echo "[dry-run] systemctl enable --now ${SERVICE_NAME}"
    if [[ "${CONFIGURE_DOCKER}" -eq 1 ]]; then
      echo "[dry-run] systemctl restart docker"
    fi
  else
    systemctl daemon-reload
    systemctl enable --now "${SERVICE_NAME}"
    if [[ "${CONFIGURE_DOCKER}" -eq 1 ]]; then
      systemctl restart docker
    fi
  fi

  cat <<EOF
Installation complete.

Service:
  systemctl status ${SERVICE_NAME}
  journalctl -u ${SERVICE_NAME} -o cat -f

Config:
  ${CONFIG_DIR}/config.yaml

Controller:
  http://127.0.0.1:${CONTROLLER_PORT}
  secret: ${secret}

If shell proxy export was enabled, reconnect your shell or run:
  source /etc/profile.d/${SERVICE_NAME}-proxy.sh
EOF
}

main "$@"
