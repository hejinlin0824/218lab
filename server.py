import http.server
import socketserver
import json
import time
import os
import urllib.parse
from threading import Lock

# ================= 1. 配置区域 =================
PORT = 8000                  
DATA_FILE = "stats.json"      # 存储访问量和点击量
PROFILES_FILE = "profiles.json" # 存储个人主页头像和简介
ACTIVE_WINDOW = 300           # 在线用户判定（秒）

# 管理员设置的账号密码
USERS = {
    "admin": "990824",
    "hejinlin": "123456",
    "student1": "123456"
}

# ================= 2. 数据处理与存储 =================
stats_lock = Lock()
profile_lock = Lock()
active_users = {} 

def load_json(filename, default_val):
    if not os.path.exists(filename):
        return default_val
    try:
        with open(filename, "r", encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return default_val

def save_json(filename, data):
    try:
        # 使用临时文件写入，防止数据损坏
        temp_file = filename + ".tmp"
        with open(temp_file, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        os.replace(temp_file, filename)
    except Exception as e:
        print(f"Error saving {filename}: {e}")

# 初始加载数据
stats_data = load_json(DATA_FILE, {"total_visits": 0, "tool_clicks": {}})
profiles_data = load_json(PROFILES_FILE, {})

# ================= 3. 核心请求处理类 =================
class LabRequestHandler(http.server.SimpleHTTPRequestHandler):
    
    # 优化：禁用 DNS 反向查询，加快局域网响应速度
    def address_string(self):
        return str(self.client_address[0])

    # 屏蔽控制台频繁的 HTTP 日志，保持界面整洁
    def log_message(self, format, *args):
        pass 

    # --- 处理所有 POST 请求 (登录、资料更新) ---
    def do_POST(self):
        if self.path == '/api/login':
            self.handle_login()
        elif self.path == '/api/profile/update':
            self.handle_profile_update()
        else:
            self.send_error(404, "API Endpoint not found")

    # --- 处理所有 GET 请求 (API 获取、静态文件) ---
    def do_GET(self):
        global stats_data, active_users, profiles_data
        
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        query = urllib.parse.parse_qs(parsed_path.query)
        client_ip = self.client_address[0]

        # 1. API: 获取实时统计
        if path == "/api/stats":
            with stats_lock:
                # 记录活跃用户心跳
                active_users[client_ip] = time.time()
                # 清理过期用户
                now = time.time()
                active_users = {ip: t for ip, t in active_users.items() if now - t < ACTIVE_WINDOW}
                
                self.send_json_response({
                    "total_visits": stats_data.get("total_visits", 0),
                    "active_users": len(active_users),
                    "tool_clicks": stats_data.get("tool_clicks", {})
                })
            return

        # 2. API: 获取个人资料 (修复 404)
        elif path == "/api/profile":
            username = query.get("user", [None])[0]
            if username:
                profile = profiles_data.get(username, {})
                self.send_json_response({
                    "status": "success",
                    "bio": profile.get("bio", "这位研究员很懒，还没有写简介。"),
                    "avatar": profile.get("avatar", "") # Base64 字符串
                })
            else:
                self.send_json_response({"status": "error", "message": "Missing user"}, 400)
            return

        # 3. API: 记录点击上报
        elif path == "/api/click":
            tool_id = query.get("id", [None])[0]
            if tool_id:
                with stats_lock:
                    clicks = stats_data.get("tool_clicks", {})
                    clicks[tool_id] = clicks.get(tool_id, 0) + 1
                    stats_data["tool_clicks"] = clicks
                    save_json(DATA_FILE, stats_data)
            self.send_response(200)
            self.end_headers()
            return

        # 4. 统计首页访问量
        if path == "/" or path == "/index.html":
            with stats_lock:
                stats_data["total_visits"] = stats_data.get("total_visits", 0) + 1
                save_json(DATA_FILE, stats_data)
        
        # 5. 返回静态文件 (index.html, icon.png 等)
        return super().do_GET()

    # --- 逻辑封装：登录 ---
    def handle_login(self):
        data = self.parse_post_data()
        if not data: return
        
        username = data.get("username")
        password = data.get("password")
        
        if username in USERS and USERS[username] == password:
            print(f"[{time.strftime('%H:%M:%S')}] ✅ Login: {username} from {self.client_address[0]}")
            self.send_json_response({
                "status": "success", 
                "user": username,
                "avatar": profiles_data.get(username, {}).get("avatar", "")
            })
        else:
            print(f"[{time.strftime('%H:%M:%S')}] ❌ Failed Login: {username}")
            self.send_json_response({"status": "error", "message": "Wrong credentials"}, 401)

    # --- 逻辑封装：更新资料 ---
    def handle_profile_update(self):
        data = self.parse_post_data()
        if not data: return
        
        username = data.get("username")
        if username and username in USERS:
            with profile_lock:
                if username not in profiles_data:
                    profiles_data[username] = {}
                
                if "bio" in data: profiles_data[username]["bio"] = data["bio"]
                if "avatar" in data and data["avatar"]: profiles_data[username]["avatar"] = data["avatar"]
                
                save_json(PROFILES_FILE, profiles_data)
            self.send_json_response({"status": "success"})
        else:
            self.send_json_response({"status": "error", "message": "Unauthorized"}, 403)

    # --- 辅助方法：发送标准 JSON 响应 ---
    def send_json_response(self, data, status=200):
        json_bytes = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Content-Length', len(json_bytes))
        self.send_header('Access-Control-Allow-Origin', '*') # 允许跨域
        self.end_headers()
        self.wfile.write(json_bytes)

    # --- 辅助方法：解析 POST 传来的 JSON 数据 ---
    def parse_post_data(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0: return None
            raw_data = self.rfile.read(content_length)
            return json.loads(raw_data.decode('utf-8'))
        except Exception as e:
            print(f"Post Data Parse Error: {e}")
            return None

# ================= 4. 启动服务 =================
if __name__ == "__main__":
    # 允许端口立即重用
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.ThreadingTCPServer(("", PORT), LabRequestHandler) as httpd:
        print(f"\n" + "="*50)
        print(f" 🚀 218 Lab Center is online at port {PORT}")
        print(f" 📂 Data Files: {DATA_FILE}, {PROFILES_FILE}")
        print(f" 🔐 Configured Users: {', '.join(USERS.keys())}")
        print(f" 💡 Press Ctrl+C to stop the server")
        print("="*50 + "\n")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server shutting down...")
            httpd.shutdown()