import http.server
import socketserver
import json
import time
import os
import urllib.parse
from threading import Thread, Lock

# ================= 配置区域 =================
PORT = 8000                  
DATA_FILE = "stats.json"     
ACTIVE_WINDOW = 300          

# ================= 全局数据与锁 =================
stats_lock = Lock()
active_users = {} 

def load_stats():
    if not os.path.exists(DATA_FILE):
        return {"total_visits": 0, "tool_clicks": {}}
    try:
        with open(DATA_FILE, "r", encoding='utf-8') as f:
            return json.load(f)
    except:
        return {"total_visits": 0, "tool_clicks": {}}

def save_stats(data):
    try:
        with open(DATA_FILE, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving stats: {e}")

stats_data = load_stats()

# ================= 请求处理逻辑 =================
class LabRequestHandler(http.server.SimpleHTTPRequestHandler):
    
    # 修复卡顿的关键：禁用反向 DNS 查询
    def address_string(self):
        return str(self.client_address[0])

    def log_message(self, format, *args):
        # 仅在有错误时打印，或者你可以注释掉 pass 来查看所有请求
        pass 

    def do_GET(self):
        global stats_data, active_users
        
        try:
            # 1. 解析请求
            parsed_path = urllib.parse.urlparse(self.path)
            path = parsed_path.path
            query = urllib.parse.parse_qs(parsed_path.query)
            client_ip = self.client_address[0]
            current_time = time.time()

            # ------------------------------------------------
            # API 1: 获取统计数据
            # ------------------------------------------------
            if path == "/api/stats":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                
                with stats_lock:
                    active_users[client_ip] = current_time
                    for ip in list(active_users.keys()):
                        if current_time - active_users[ip] > ACTIVE_WINDOW:
                            del active_users[ip]
                    
                    response_data = {
                        "total_visits": stats_data.get("total_visits", 0),
                        "active_users": len(active_users),
                        "tool_clicks": stats_data.get("tool_clicks", {})
                    }
                
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
                return

            # ------------------------------------------------
            # API 2: 记录工具点击
            # ------------------------------------------------
            elif path == "/api/click":
                tool_id = query.get("id", [None])[0]
                if tool_id:
                    with stats_lock:
                        clicks = stats_data.get("tool_clicks", {})
                        clicks[tool_id] = clicks.get(tool_id, 0) + 1
                        stats_data["tool_clicks"] = clicks
                        save_stats(stats_data)
                        print(f"[{time.strftime('%H:%M:%S')}] User {client_ip} clicked: {tool_id}")

                self.send_response(200)
                self.end_headers()
                return

            # ------------------------------------------------
            # 页面访问统计
            # ------------------------------------------------
            if path == "/" or path == "/index.html":
                with stats_lock:
                    stats_data["total_visits"] = stats_data.get("total_visits", 0) + 1
                    save_stats(stats_data)
                    print(f"[{time.strftime('%H:%M:%S')}] Visit from {client_ip} | Total: {stats_data['total_visits']}")

            # ------------------------------------------------
            # 静态文件服务
            # ------------------------------------------------
            return super().do_GET()
            
        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception as e:
            print(f"Request Error: {e}")

# ================= 启动服务 =================
if __name__ == "__main__":
    with socketserver.ThreadingTCPServer(("", PORT), LabRequestHandler) as httpd:
        print(f"\n==================================================")
        print(f" 🚀 218 Lab Server Running on Port {PORT}")
        print(f" ⚡ Speed optimized (DNS lookup disabled)")
        print(f"==================================================\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            httpd.shutdown()