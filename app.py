import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import base64
import io
from PIL import Image
import json
import sqlite3
from datetime import datetime
import os
import threading
import time
import face_recognition
import pickle
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# 初始化MediaPipe人脸检测
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)

# 人脸识别相关变量
known_face_encodings = []
known_face_names = []
face_tracking = {}  # 跟踪每个人脸的状态
person_appearances = defaultdict(list)  # 记录每个人的出现时间
TRACKING_TIMEOUT = 3  # 3秒没检测到就认为离开了

# 初始化数据库
def init_db():
    """初始化SQLite数据库"""
    conn = sqlite3.connect('face_records.db')
    c = conn.cursor()
    
    # 创建人脸注册表
    c.execute('''CREATE TABLE IF NOT EXISTS registered_faces
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT UNIQUE,
                  encoding BLOB,
                  photo_path TEXT,
                  created_at TEXT)''')
    
    # 创建出现记录表
    c.execute('''CREATE TABLE IF NOT EXISTS appearance_records
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  person_name TEXT,
                  start_time TEXT,
                  end_time TEXT,
                  duration REAL,
                  confidence REAL)''')
    
    # 创建统计表
    c.execute('''CREATE TABLE IF NOT EXISTS person_statistics
                 (person_name TEXT PRIMARY KEY,
                  total_appearances INTEGER,
                  total_duration REAL,
                  last_seen TEXT,
                  first_seen TEXT)''')
    
    conn.commit()
    conn.close()
    
    # 加载已注册的人脸
    load_registered_faces()

def load_registered_faces():
    """从数据库加载已注册的人脸"""
    global known_face_encodings, known_face_names
    
    conn = sqlite3.connect('face_records.db')
    c = conn.cursor()
    c.execute("SELECT name, encoding FROM registered_faces")
    
    known_face_encodings = []
    known_face_names = []
    
    for name, encoding_blob in c.fetchall():
        encoding = pickle.loads(encoding_blob)
        known_face_encodings.append(encoding)
        known_face_names.append(name)
    
    conn.close()
    print(f"已加载 {len(known_face_names)} 个注册人脸")

def update_person_statistics(person_name, start_time, end_time):
    """更新人员统计信息"""
    conn = sqlite3.connect('face_records.db')
    c = conn.cursor()
    
    duration = (end_time - start_time).total_seconds()
    
    # 检查是否已有统计记录
    c.execute("SELECT total_appearances, total_duration FROM person_statistics WHERE person_name = ?", 
              (person_name,))
    result = c.fetchone()
    
    if result:
        # 更新现有记录
        total_appearances, total_duration = result
        c.execute("""UPDATE person_statistics 
                     SET total_appearances = ?, total_duration = ?, last_seen = ?
                     WHERE person_name = ?""",
                  (total_appearances + 1, total_duration + duration, 
                   end_time.isoformat(), person_name))
    else:
        # 创建新记录
        c.execute("""INSERT INTO person_statistics 
                     (person_name, total_appearances, total_duration, first_seen, last_seen)
                     VALUES (?, ?, ?, ?, ?)""",
                  (person_name, 1, duration, start_time.isoformat(), end_time.isoformat()))
    
    conn.commit()
    conn.close()

# HTML模板
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>智能人脸识别考勤系统</title>
    <meta charset="utf-8">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .panel {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        }
        
        .full-width {
            grid-column: 1 / -1;
        }
        
        #videoContainer {
            position: relative;
            width: 100%;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        
        #video, #canvas {
            width: 100%;
            display: block;
        }
        
        #canvas {
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
        }
        
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 30px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .register-form {
            display: none;
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #333;
        }
        
        input[type="text"] {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        
        input[type="text"]:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }
        
        .stat-label {
            color: #666;
            font-size: 14px;
            margin-bottom: 10px;
        }
        
        .stat-value {
            font-size: 28px;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .person-list {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .person-item {
            padding: 15px;
            margin-bottom: 10px;
            background: #f8f9fa;
            border-radius: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.3s ease;
        }
        
        .person-item:hover {
            transform: translateX(5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .person-info {
            flex: 1;
        }
        
        .person-name {
            font-weight: bold;
            font-size: 18px;
            margin-bottom: 5px;
        }
        
        .person-stats {
            color: #666;
            font-size: 14px;
        }
        
        .active-person {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0, 255, 0, 0.9);
            color: white;
            padding: 10px 20px;
            border-radius: 20px;
            font-weight: bold;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 0.8; }
            50% { opacity: 1; }
            100% { opacity: 0.8; }
        }
        
        .delete-btn {
            background: #dc3545;
            padding: 5px 15px;
            font-size: 14px;
        }
        
        .delete-btn:hover {
            background: #c82333;
        }
        
        #status {
            background: #e3f2fd;
            color: #1976d2;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-weight: 500;
            margin-bottom: 20px;
        }
        
        .registered-faces {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .face-card {
            text-align: center;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        
        .face-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .face-photo {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            object-fit: cover;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 智能人脸识别考勤系统</h1>
        
        <div class="main-grid">
            <!-- 左侧：视频和控制 -->
            <div class="panel">
                <h2>实时监控</h2>
                <div id="videoContainer">
                    <video id="video" autoplay muted></video>
                    <canvas id="canvas"></canvas>
                    <div id="activePerson" class="active-person" style="display: none;"></div>
                </div>
                
                <div class="controls">
                    <button id="startBtn" onclick="startCamera()">
                        <span>🎥</span> 开始识别
                    </button>
                    <button id="stopBtn" onclick="stopCamera()" disabled>
                        <span>⏹️</span> 停止识别
                    </button>
                    <button id="registerBtn" onclick="toggleRegister()">
                        <span>➕</span> 注册新人脸
                    </button>
                    <button onclick="exportData()">
                        <span>💾</span> 导出数据
                    </button>
                </div>
                
                <div id="registerForm" class="register-form">
                    <h3>注册新人脸</h3>
                    <div class="form-group">
                        <label for="personName">姓名：</label>
                        <input type="text" id="personName" placeholder="请输入姓名">
                    </div>
                    <button onclick="captureFace()">📸 拍照注册</button>
                    <button onclick="toggleRegister()">取消</button>
                </div>
                
                <div id="status">系统就绪，等待开始...</div>
            </div>
            
            <!-- 右侧：统计信息 -->
            <div class="panel">
                <h2>今日统计</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">已注册人数</div>
                        <div class="stat-value" id="registeredCount">0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">今日签到</div>
                        <div class="stat-value" id="todayCount">0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">当前在场</div>
                        <div class="stat-value" id="currentCount">0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">平均停留</div>
                        <div class="stat-value" id="avgDuration">0分</div>
                    </div>
                </div>
                
                <h3 style="margin-top: 20px;">已注册人员</h3>
                <div id="registeredFaces" class="registered-faces"></div>
            </div>
            
            <!-- 底部：人员列表 -->
            <div class="panel full-width">
                <h2>考勤记录</h2>
                <div id="personList" class="person-list"></div>
            </div>
        </div>
    </div>

    <script>
        let video = document.getElementById('video');
        let canvas = document.getElementById('canvas');
        let ctx = canvas.getContext('2d');
        let stream = null;
        let detecting = false;
        let registerMode = false;
        let currentDetections = new Map();  // 当前检测到的人脸
        
        // 按钮元素
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const registerBtn = document.getElementById('registerBtn');

        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: 'user'
                    } 
                });
                
                video.srcObject = stream;
                video.onloadedmetadata = () => {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    detecting = true;
                    detectAndRecognize();
                    updateStatus('正在实时识别...', '#4caf50');
                    
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                };
                
                // 加载统计信息
                loadStatistics();
                
            } catch (err) {
                updateStatus('无法访问摄像头: ' + err.message, '#f44336');
            }
        }

        function stopCamera() {
            detecting = false;
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            updateStatus('识别已停止', '#ff9800');
            document.getElementById('activePerson').style.display = 'none';
            
            startBtn.disabled = false;
            stopBtn.disabled = true;
        }

        async function detectAndRecognize() {
            if (!detecting) return;

            // 捕获当前帧
            let tempCanvas = document.createElement('canvas');
            tempCanvas.width = video.videoWidth;
            tempCanvas.height = video.videoHeight;
            let tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(video, 0, 0);
            
            let imageData = tempCanvas.toDataURL('image/jpeg', 0.8);
            
            try {
                let response = await fetch('/recognize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ image: imageData })
                });
                
                let result = await response.json();
                
                // 清除画布
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // 处理识别结果
                let detectedNow = new Set();
                
                result.faces.forEach(face => {
                    let bbox = face.bbox;
                    
                    // 绘制边界框
                    if (face.name === 'Unknown') {
                        ctx.strokeStyle = '#ff0000';  // 红色表示未识别
                    } else {
                        ctx.strokeStyle = '#00ff00';  // 绿色表示已识别
                        detectedNow.add(face.name);
                    }
                    
                    ctx.lineWidth = 3;
                    ctx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);
                    
                    // 绘制标签
                    ctx.fillStyle = ctx.strokeStyle;
                    ctx.fillRect(bbox.x, bbox.y - 30, bbox.width, 30);
                    ctx.fillStyle = 'white';
                    ctx.font = 'bold 16px Arial';
                    ctx.fillText(face.name, bbox.x + 5, bbox.y - 8);
                });
                
                // 更新活跃人员显示
                if (detectedNow.size > 0) {
                    let names = Array.from(detectedNow).join(', ');
                    document.getElementById('activePerson').textContent = '当前: ' + names;
                    document.getElementById('activePerson').style.display = 'block';
                } else {
                    document.getElementById('activePerson').style.display = 'none';
                }
                
                // 更新人员状态
                updatePersonTracking(detectedNow);
                
                // 更新统计
                updateStatistics();
                
            } catch (err) {
                console.error('识别错误:', err);
            }
            
            // 继续下一帧
            setTimeout(detectAndRecognize, 100);
        }

        function updatePersonTracking(detectedNow) {
            // 检查新出现的人
            detectedNow.forEach(name => {
                if (!currentDetections.has(name)) {
                    // 新检测到的人
                    currentDetections.set(name, {
                        firstSeen: new Date(),
                        lastSeen: new Date()
                    });
                    console.log(`${name} 进入画面`);
                } else {
                    // 更新最后看到的时间
                    currentDetections.get(name).lastSeen = new Date();
                }
            });
            
            // 检查离开的人
            currentDetections.forEach((info, name) => {
                if (!detectedNow.has(name)) {
                    // 检查是否超时
                    let timeSinceLastSeen = (new Date() - info.lastSeen) / 1000;
                    if (timeSinceLastSeen > 3) {  // 3秒超时
                        // 记录这次出现
                        recordAppearance(name, info.firstSeen, info.lastSeen);
                        currentDetections.delete(name);
                        console.log(`${name} 离开画面`);
                    }
                }
            });
        }

        async function recordAppearance(name, startTime, endTime) {
            try {
                await fetch('/record_appearance', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        name: name,
                        start_time: startTime.toISOString(),
                        end_time: endTime.toISOString()
                    })
                });
                
                // 刷新统计信息
                loadStatistics();
                
            } catch (err) {
                console.error('记录失败:', err);
            }
        }

        function toggleRegister() {
            let form = document.getElementById('registerForm');
            registerMode = !registerMode;
            form.style.display = registerMode ? 'block' : 'none';
        }

        async function captureFace() {
            let name = document.getElementById('personName').value.trim();
            if (!name) {
                alert('请输入姓名');
                return;
            }
            
            if (!stream) {
                alert('请先开启摄像头');
                return;
            }
            
            // 捕获当前帧
            let tempCanvas = document.createElement('canvas');
            tempCanvas.width = video.videoWidth;
            tempCanvas.height = video.videoHeight;
            let tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(video, 0, 0);
            
            let imageData = tempCanvas.toDataURL('image/jpeg', 0.9);
            
            try {
                let response = await fetch('/register_face', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        name: name,
                        image: imageData
                    })
                });
                
                let result = await response.json();
                
                if (result.success) {
                    updateStatus(`✅ 成功注册 ${name}！`, '#4caf50');
                    document.getElementById('personName').value = '';
                    toggleRegister();
                    loadRegisteredFaces();
                } else {
                    updateStatus(`❌ 注册失败: ${result.message}`, '#f44336');
                }
                
            } catch (err) {
                updateStatus('❌ 注册失败: ' + err.message, '#f44336');
            }
        }

        async function loadStatistics() {
            try {
                let response = await fetch('/statistics');
                let data = await response.json();
                
                document.getElementById('registeredCount').textContent = data.registered_count;
                document.getElementById('todayCount').textContent = data.today_count;
                document.getElementById('currentCount').textContent = currentDetections.size;
                document.getElementById('avgDuration').textContent = data.avg_duration + '分';
                
                // 更新人员列表
                updatePersonList(data.person_stats);
                
            } catch (err) {
                console.error('加载统计失败:', err);
            }
        }

        async function loadRegisteredFaces() {
            try {
                let response = await fetch('/registered_faces');
                let data = await response.json();
                
                let container = document.getElementById('registeredFaces');
                container.innerHTML = data.faces.map(face => `
                    <div class="face-card">
                        <img src="${face.photo}" class="face-photo" alt="${face.name}">
                        <div>${face.name}</div>
                        <button class="delete-btn" onclick="deleteFace('${face.name}')">删除</button>
                    </div>
                `).join('');
                
            } catch (err) {
                console.error('加载注册人脸失败:', err);
            }
        }

        async function deleteFace(name) {
            if (!confirm(`确定要删除 ${name} 吗？`)) return;
            
            try {
                let response = await fetch('/delete_face', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ name: name })
                });
                
                if (response.ok) {
                    updateStatus(`已删除 ${name}`, '#ff9800');
                    loadRegisteredFaces();
                    loadStatistics();
                }
                
            } catch (err) {
                console.error('删除失败:', err);
            }
        }

        function updatePersonList(personStats) {
            let list = document.getElementById('personList');
            list.innerHTML = personStats.map(person => `
                <div class="person-item">
                    <div class="person-info">
                        <div class="person-name">${person.name}</div>
                        <div class="person-stats">
                            出现次数: ${person.appearances} | 
                            总时长: ${person.total_duration}分钟 | 
                            最后出现: ${person.last_seen}
                        </div>
                    </div>
                </div>
            `).join('');
        }

        function updateStatistics() {
            document.getElementById('currentCount').textContent = currentDetections.size;
        }

        function updateStatus(message, color = '#1976d2') {
            const status = document.getElementById('status');
            status.textContent = message;
            status.style.background = color + '20';
            status.style.color = color;
        }

        async function exportData() {
            try {
                let response = await fetch('/export_data');
                let blob = await response.blob();
                
                let url = URL.createObjectURL(blob);
                let a = document.createElement('a');
                a.href = url;
                a.download = `attendance_${new Date().toISOString().slice(0,10)}.json`;
                a.click();
                
                URL.revokeObjectURL(url);
                updateStatus('✅ 数据导出成功！', '#4caf50');
                
            } catch (err) {
                updateStatus('❌ 导出失败: ' + err.message, '#f44336');
            }
        }

        // 页面加载时初始化
        window.onload = () => {
            loadStatistics();
            loadRegisteredFaces();
            // 定期刷新统计
            setInterval(loadStatistics, 5000);
        };

        // 页面关闭时清理
        window.onbeforeunload = () => {
            if (stream) stopCamera();
        };
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """返回主页面"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/recognize', methods=['POST'])
def recognize_faces():
    """人脸识别接口"""
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        # 转换为numpy数组
        img_array = np.array(image)
        
        # 使用face_recognition检测和识别
        face_locations = face_recognition.face_locations(img_array)
        face_encodings = face_recognition.face_encodings(img_array, face_locations)
        
        faces = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # 默认为未知
            name = "Unknown"
            confidence = 0
            
            if known_face_encodings:
                # 计算与已知人脸的距离
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    min_distance = face_distances[best_match_index]
                    
                    # 如果距离小于阈值，认为是匹配的
                    if min_distance < 0.6:  # 可调整阈值
                        name = known_face_names[best_match_index]
                        confidence = 1 - min_distance
            
            faces.append({
                'bbox': {
                    'x': left,
                    'y': top,
                    'width': right - left,
                    'height': bottom - top
                },
                'name': name,
                'confidence': confidence
            })
        
        return jsonify({'faces': faces})
    
    except Exception as e:
        app.logger.error(f"Recognition error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/register_face', methods=['POST'])
def register_face():
    """注册新人脸"""
    try:
        data = request.json
        name = data['name']
        image_data = data['image'].split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        # 转换为numpy数组
        img_array = np.array(image)
        
        # 检测人脸
        face_locations = face_recognition.face_locations(img_array)
        
        if len(face_locations) == 0:
            return jsonify({'success': False, 'message': '未检测到人脸'})
        
        if len(face_locations) > 1:
            return jsonify({'success': False, 'message': '检测到多张人脸，请确保只有一个人'})
        
        # 提取人脸特征
        face_encodings = face_recognition.face_encodings(img_array, face_locations)
        face_encoding = face_encodings[0]
        
        # 保存到数据库
        conn = sqlite3.connect('face_records.db')
        c = conn.cursor()
        
        # 检查是否已存在
        c.execute("SELECT id FROM registered_faces WHERE name = ?", (name,))
        if c.fetchone():
            conn.close()
            return jsonify({'success': False, 'message': '该姓名已存在'})
        
        # 保存图片
        os.makedirs('registered_faces', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        photo_path = f'registered_faces/{name}_{timestamp}.jpg'
        image.save(photo_path)
        
        # 保存到数据库
        encoding_blob = pickle.dumps(face_encoding)
        c.execute("""INSERT INTO registered_faces (name, encoding, photo_path, created_at)
                     VALUES (?, ?, ?, ?)""",
                  (name, encoding_blob, photo_path, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        # 更新内存中的人脸数据
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)
        
        return jsonify({'success': True, 'message': f'成功注册 {name}'})
    
    except Exception as e:
        app.logger.error(f"Registration error: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/record_appearance', methods=['POST'])
def record_appearance():
    """记录人员出现"""
    try:
        data = request.json
        name = data['name']
        start_time = datetime.fromisoformat(data['start_time'].replace('Z', '+00:00'))
        end_time = datetime.fromisoformat(data['end_time'].replace('Z', '+00:00'))
        
        duration = (end_time - start_time).total_seconds()
        
        conn = sqlite3.connect('face_records.db')
        c = conn.cursor()
        
        # 记录这次出现
        c.execute("""INSERT INTO appearance_records 
                     (person_name, start_time, end_time, duration, confidence)
                     VALUES (?, ?, ?, ?, ?)""",
                  (name, start_time.isoformat(), end_time.isoformat(), duration, 0.95))
        
        conn.commit()
        conn.close()
        
        # 更新统计信息
        update_person_statistics(name, start_time, end_time)
        
        return jsonify({'success': True})
    
    except Exception as e:
        app.logger.error(f"Record error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/statistics', methods=['GET'])
def get_statistics():
    """获取统计信息"""
    try:
        conn = sqlite3.connect('face_records.db')
        c = conn.cursor()
        
        # 获取注册人数
        c.execute("SELECT COUNT(*) FROM registered_faces")
        registered_count = c.fetchone()[0]
        
        # 获取今日签到人数
        today = datetime.now().date().isoformat()
        c.execute("""SELECT COUNT(DISTINCT person_name) FROM appearance_records 
                     WHERE DATE(start_time) = ?""", (today,))
        today_count = c.fetchone()[0]
        
        # 获取平均停留时间（分钟）
        c.execute("SELECT AVG(duration) FROM appearance_records WHERE DATE(start_time) = ?", (today,))
        avg_duration = c.fetchone()[0] or 0
        avg_duration = round(avg_duration / 60, 1)  # 转换为分钟
        
        # 获取每个人的统计信息
        c.execute("""SELECT ps.*, 
                     (SELECT COUNT(*) FROM appearance_records 
                      WHERE person_name = ps.person_name AND DATE(start_time) = ?) as today_count
                     FROM person_statistics ps
                     ORDER BY ps.last_seen DESC""", (today,))
        
        person_stats = []
        for row in c.fetchall():
            person_stats.append({
                'name': row[0],
                'appearances': row[1],
                'total_duration': round(row[2] / 60, 1),  # 转换为分钟
                'last_seen': datetime.fromisoformat(row[3]).strftime('%Y-%m-%d %H:%M'),
                'today_count': row[5]
            })
        
        conn.close()
        
        return jsonify({
            'registered_count': registered_count,
            'today_count': today_count,
            'avg_duration': avg_duration,
            'person_stats': person_stats
        })
    
    except Exception as e:
        app.logger.error(f"Statistics error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/registered_faces', methods=['GET'])
def get_registered_faces():
    """获取已注册的人脸列表"""
    try:
        conn = sqlite3.connect('face_records.db')
        c = conn.cursor()
        
        c.execute("SELECT name, photo_path FROM registered_faces ORDER BY created_at DESC")
        
        faces = []
        for name, photo_path in c.fetchall():
            # 读取图片并转换为base64
            if os.path.exists(photo_path):
                with open(photo_path, 'rb') as f:
                    photo_data = base64.b64encode(f.read()).decode()
                    faces.append({
                        'name': name,
                        'photo': f'data:image/jpeg;base64,{photo_data}'
                    })
        
        conn.close()
        
        return jsonify({'faces': faces})
    
    except Exception as e:
        app.logger.error(f"Get faces error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/delete_face', methods=['POST'])
def delete_face():
    """删除注册的人脸"""
    try:
        data = request.json
        name = data['name']
        
        conn = sqlite3.connect('face_records.db')
        c = conn.cursor()
        
        # 获取照片路径
        c.execute("SELECT photo_path FROM registered_faces WHERE name = ?", (name,))
        result = c.fetchone()
        
        if result:
            photo_path = result[0]
            # 删除照片文件
            if os.path.exists(photo_path):
                os.remove(photo_path)
            
            # 从数据库删除
            c.execute("DELETE FROM registered_faces WHERE name = ?", (name,))
            
            # 删除相关记录
            c.execute("DELETE FROM appearance_records WHERE person_name = ?", (name,))
            c.execute("DELETE FROM person_statistics WHERE person_name = ?", (name,))
            
            conn.commit()
        
        conn.close()
        
        # 重新加载人脸数据
        load_registered_faces()
        
        return jsonify({'success': True})
    
    except Exception as e:
        app.logger.error(f"Delete error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/export_data', methods=['GET'])
def export_data():
    """导出数据"""
    try:
        conn = sqlite3.connect('face_records.db')
        c = conn.cursor()
        
        # 获取所有数据
        data = {
            'export_time': datetime.now().isoformat(),
            'registered_faces': [],
            'appearance_records': [],
            'statistics': []
        }
        
        # 注册人脸
        c.execute("SELECT name, created_at FROM registered_faces")
        for row in c.fetchall():
            data['registered_faces'].append({
                'name': row[0],
                'created_at': row[1]
            })
        
        # 出现记录
        c.execute("SELECT * FROM appearance_records ORDER BY start_time DESC")
        for row in c.fetchall():
            data['appearance_records'].append({
                'person_name': row[1],
                'start_time': row[2],
                'end_time': row[3],
                'duration': row[4]
            })
        
        # 统计信息
        c.execute("SELECT * FROM person_statistics")
        for row in c.fetchall():
            data['statistics'].append({
                'person_name': row[0],
                'total_appearances': row[1],
                'total_duration': row[2],
                'first_seen': row[4],
                'last_seen': row[3]
            })
        
        conn.close()
        
        # 返回JSON文件
        response = app.response_class(
            response=json.dumps(data, indent=2, ensure_ascii=False),
            status=200,
            mimetype='application/json'
        )
        response.headers['Content-Disposition'] = f'attachment; filename=attendance_{datetime.now().strftime("%Y%m%d")}.json'
        
        return response
    
    except Exception as e:
        app.logger.error(f"Export error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'registered_faces': len(known_face_names)
    })

if __name__ == '__main__':
    # 创建必要的目录
    os.makedirs('captures', exist_ok=True)
    os.makedirs('registered_faces', exist_ok=True)
    
    # 初始化数据库
    init_db()
    
    # 配置日志
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 启动应用
    app.run(host='0.0.0.0', port=5000, debug=False)