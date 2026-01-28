from flask import Flask, render_template_string, request, session, redirect, jsonify
import sqlite3, hashlib, jwt, pickle, base64, os, re
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = 'ctf_secret_key_change_this'

# Database setup
def init_db():
    conn = sqlite3.connect('ctf.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS progress 
                 (id INTEGER PRIMARY KEY, challenge_id INTEGER, solved INTEGER, hints_used INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY, username TEXT, password TEXT, role TEXT, api_key TEXT)''')
    c.execute("INSERT OR IGNORE INTO users VALUES (1, 'admin', 'admin123', 'admin', 'secret_api_key_12345')")
    c.execute("INSERT OR IGNORE INTO users VALUES (2, 'guest', 'guest', 'user', 'guest_key_67890')")
    conn.commit()
    conn.close()

init_db()

# Challenges configuration
CHALLENGES = {
    1: {
        'name': 'JWT Algorithm Confusion',
        'desc': 'The API uses JWT for authentication. Can you become admin?',
        'difficulty': 'Easy',
        'category': 'Authentication',
        'flag': 'CTF{alg_none_is_dangerous}',
        'hints': ['Check the JWT algorithm', 'Try changing alg to "none"', 'Remove the signature part'],
        'solution': 'Decode JWT, change alg to "none", set role to admin, remove signature'
    },
    2: {
        'name': 'SSTI Paradise',
        'desc': 'Search for users in our system. Template: {{username}}',
        'difficulty': 'Medium',
        'category': 'Server-Side',
        'flag': 'CTF{jinja2_rce_achieved}',
        'hints': ['Templates can execute code', 'Try {{7*7}}', 'Use __class__.__mro__ for RCE'],
        'solution': 'Use {{config}} or {{self.__dict__}} to leak flag variable'
    },
    3: {
        'name': 'Prototype Pollution',
        'desc': 'Our JSON merger is perfectly safe... or is it?',
        'difficulty': 'Hard',
        'category': 'Client-Side',
        'flag': 'CTF{polluted_proto_chain}',
        'hints': ['__proto__ is special in JS', 'Merge can modify Object.prototype', 'Check constructor.prototype'],
        'solution': 'Send {"__proto__": {"isAdmin": true}} to pollute prototype'
    },
    4: {
        'name': 'GraphQL Introspection',
        'desc': 'Our GraphQL API is production-ready with introspection disabled... right?',
        'difficulty': 'Easy',
        'category': 'API Security',
        'flag': 'CTF{graphql_introspection_exposed}',
        'hints': ['Try __schema query', 'Batch queries might work', 'Check for query depth limits'],
        'solution': 'Use __schema{types{name,fields{name}}} to discover hidden queries'
    },
    5: {
        'name': 'Pickle Deserialization',
        'desc': 'Save your session data securely with pickle!',
        'difficulty': 'Hard',
        'category': 'Server-Side',
        'flag': 'CTF{pickles_are_not_secure}',
        'hints': ['Pickle can execute code', 'Use __reduce__ method', 'Create malicious pickle object'],
        'solution': 'Create pickle payload with __reduce__ to execute system commands'
    },
    6: {
        'name': 'Race Condition Bank',
        'desc': 'Transfer money between accounts. Balance checks are atomic... we think.',
        'difficulty': 'Medium',
        'category': 'Business Logic',
        'flag': 'CTF{race_to_the_bank}',
        'hints': ['Send multiple requests simultaneously', 'Check balance validation timing', 'Use threading or async'],
        'solution': 'Send parallel withdrawal requests before balance updates'
    },
    7: {
        'name': 'SSRF Cloud Metadata',
        'desc': 'Check if a URL is alive. Internal networks are blocked!',
        'difficulty': 'Medium',
        'category': 'Server-Side',
        'flag': 'CTF{cloud_metadata_leaked}',
        'hints': ['169.254.169.254 is special', 'Try URL encoding', 'DNS rebinding might help'],
        'solution': 'Access http://169.254.169.254/latest/meta-data/ via SSRF'
    },
    8: {
        'name': 'OAuth Token Confusion',
        'desc': 'Login with OAuth. Tokens are validated... somewhat.',
        'difficulty': 'Hard',
        'category': 'Authentication',
        'flag': 'CTF{oauth_flow_broken}',
        'hints': ['Check redirect_uri validation', 'CSRF token missing?', 'Try token replay'],
        'solution': 'Intercept OAuth callback and reuse access_token for different user'
    },
    9: {
        'name': 'DOM Clobbering',
        'desc': 'Our XSS filters are top-notch! No script tags allowed.',
        'difficulty': 'Medium',
        'category': 'Client-Side',
        'flag': 'CTF{clobbered_the_dom}',
        'hints': ['HTML id/name can override JS', 'Form elements create globals', 'Use anchor tags with name'],
        'solution': 'Use <a id="config" href="javascript:alert()">to clobber config object'
    },
    10: {
        'name': 'SQL Injection - WAF Bypass',
        'desc': 'Search users. Our WAF blocks common SQL injection keywords!',
        'difficulty': 'Hard',
        'category': 'Database',
        'flag': 'CTF{waf_bypass_master}',
        'hints': ['Try alternative SQL syntax', 'Comments can hide keywords', 'Use UNION with NULL padding'],
        'solution': 'Use /*!50000UNION*/ or hex encoding to bypass WAF filters'
    }
}

# HTML Templates
MAIN_TEMPLATE = '''
<!DOCTYPE html>
<html><head><title>Advanced Web CTF</title>
<style>
body{font-family:Arial;margin:0;padding:20px;background:#1a1a2e;color:#eee}
.container{max-width:1200px;margin:0 auto}
h1{color:#16db93;text-align:center}
.challenges{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:20px;margin-top:30px}
.challenge{background:#16213e;padding:20px;border-radius:8px;border-left:4px solid #16db93;cursor:pointer;transition:transform 0.2s}
.challenge:hover{transform:translateY(-5px)}
.difficulty{display:inline-block;padding:4px 12px;border-radius:12px;font-size:12px;font-weight:bold}
.Easy{background:#16db93;color:#000}
.Medium{background:#f39c12;color:#000}
.Hard{background:#e74c3c;color:#fff}
.category{color:#16db93;font-size:14px;margin:10px 0}
.solved{opacity:0.6;border-left-color:#888}
h2{margin-top:0}
</style>
</head><body>
<div class="container">
<h1> Advanced Web Security CTF</h1>
<p style="text-align:center">10 Advanced Web Application Challenges | No PortSwigger Duplicates</p>
<div class="challenges">
{% for id, ch in challenges.items() %}
<div class="challenge {{ 'solved' if solved[id] else '' }}" onclick="location.href='/challenge/{{id}}'">
<h2>{{ch.name}} {% if solved[id] %}✓{% endif %}</h2>
<div class="category">{{ch.category}}</div>
<span class="difficulty {{ch.difficulty}}">{{ch.difficulty}}</span>
<p>{{ch.desc}}</p>
</div>
{% endfor %}
</div>
</div>
</body></html>
'''

CHALLENGE_TEMPLATE = '''
<!DOCTYPE html>
<html><head><title>{{ch.name}} - CTF</title>
<style>
body{font-family:Arial;margin:0;padding:20px;background:#1a1a2e;color:#eee}
.container{max-width:800px;margin:0 auto}
.back{color:#16db93;text-decoration:none;font-size:20px}
.challenge-header{background:#16213e;padding:30px;border-radius:8px;margin:20px 0}
.difficulty{display:inline-block;padding:4px 12px;border-radius:12px;font-size:12px;font-weight:bold}
.Easy{background:#16db93;color:#000}
.Medium{background:#f39c12;color:#000}
.Hard{background:#e74c3c;color:#fff}
.section{background:#16213e;padding:20px;border-radius:8px;margin:20px 0}
.hint{background:#0f3460;padding:15px;margin:10px 0;border-radius:4px;cursor:pointer}
.hint:hover{background:#1a4d7a}
input[type="text"]{width:calc(100% - 100px);padding:12px;background:#0f3460;border:1px solid #16db93;color:#eee;border-radius:4px;font-size:16px}
button{padding:12px 30px;background:#16db93;border:none;border-radius:4px;cursor:pointer;font-size:16px;font-weight:bold}
button:hover{background:#13b87a}
.success{color:#16db93;font-weight:bold;font-size:18px}
.error{color:#e74c3c;font-weight:bold}
.solution{background:#0f3460;padding:20px;border-radius:8px;white-space:pre-wrap}
#lab{background:#16213e;padding:20px;border-radius:8px;min-height:200px;margin:20px 0}
</style>
</head><body>
<div class="container">
<a href="/" class="back">← Back to Challenges</a>
<div class="challenge-header">
<h1>{{ch.name}}</h1>
<span class="difficulty {{ch.difficulty}}">{{ch.difficulty}}</span>
<span style="color:#16db93;margin-left:15px">{{ch.category}}</span>
<p style="font-size:18px;margin-top:20px">{{ch.desc}}</p>
</div>

<div class="section">
<h2> Challenge Lab</h2>
<div id="lab">
{% if id == 1 %}
<p>API Endpoint: <code>/api/verify-token</code></p>
<p>Send POST with JWT token in Authorization header</p>
<input type="text" id="jwt" placeholder="Bearer eyJ..." style="width:90%">
<button onclick="verifyJWT()">Verify Token</button>
<div id="result"></div>
{% elif id == 2 %}
<p>Search Users:</p>
<input type="text" id="search" placeholder="Enter username template">
<button onclick="searchUser()">Search</button>
<div id="result"></div>
{% elif id == 3 %}
<p>JSON Merge API:</p>
<textarea id="json" style="width:90%;height:100px;background:#0f3460;color:#eee;border:1px solid #16db93;padding:10px">{"name":"test"}</textarea>
<button onclick="mergeJSON()">Merge Object</button>
<div id="result"></div>
{% elif id == 4 %}
<p>GraphQL Endpoint: <code>/graphql</code></p>
<textarea id="query" style="width:90%;height:100px;background:#0f3460;color:#eee;border:1px solid #16db93;padding:10px">{ users { name } }</textarea>
<button onclick="runGraphQL()">Run Query</button>
<div id="result"></div>
{% elif id == 5 %}
<p>Session Serializer:</p>
<input type="text" id="data" placeholder="Enter session data">
<button onclick="saveSession()">Save Session</button>
<div id="result"></div>
{% elif id == 6 %}
<p>Bank Transfer (Balance: $1000):</p>
<input type="number" id="amount" placeholder="Amount" style="width:150px">
<button onclick="transfer()">Transfer</button>
<div id="result"></div>
{% elif id == 7 %}
<p>URL Health Checker:</p>
<input type="text" id="url" placeholder="http://example.com">
<button onclick="checkURL()">Check URL</button>
<div id="result"></div>
{% elif id == 8 %}
<p>OAuth Login Simulator:</p>
<button onclick="startOAuth()">Login with OAuth</button>
<div id="result"></div>
{% elif id == 9 %}
<p>Comment System (No XSS!):</p>
<input type="text" id="comment" placeholder="Enter comment">
<button onclick="postComment()">Post</button>
<div id="result"></div>
{% elif id == 10 %}
<p>User Search (WAF Protected):</p>
<input type="text" id="query" placeholder="Enter search term">
<button onclick="searchSQL()">Search</button>
<div id="result"></div>
{% endif %}
</div>
</div>

<div class="section">
<h2> Hints ({{hints_used}}/{{ch.hints|length}} used)</h2>
{% for i, hint in enumerate(ch.hints) %}
<div class="hint" onclick="showHint({{id}}, {{i}})">
Hint {{i+1}}: {% if i < hints_used %}{{hint}}{% else %}Click to reveal{% endif %}
</div>
{% endfor %}
</div>

<div class="section">
<h2> Submit Flag</h2>
<form method="POST">
<input type="text" name="flag" placeholder="CTF{...}" required>
<button type="submit">Submit</button>
</form>
{% if message %}<p class="{{ 'success' if success else 'error' }}">{{message}}</p>{% endif %}
</div>

{% if solved %}
<div class="section">
<h2> Solution</h2>
<div class="solution">{{ch.solution}}</div>
</div>
{% endif %}
</div>

<script>
function verifyJWT(){fetch('/api/verify-token',{method:'POST',headers:{'Authorization':document.getElementById('jwt').value}}).then(r=>r.json()).then(d=>document.getElementById('result').innerHTML='<p>'+JSON.stringify(d)+'</p>')}
function searchUser(){fetch('/api/search?q='+encodeURIComponent(document.getElementById('search').value)).then(r=>r.text()).then(d=>document.getElementById('result').innerHTML='<p>'+d+'</p>')}
function mergeJSON(){fetch('/api/merge',{method:'POST',headers:{'Content-Type':'application/json'},body:document.getElementById('json').value}).then(r=>r.json()).then(d=>document.getElementById('result').innerHTML='<p>'+JSON.stringify(d)+'</p>')}
function runGraphQL(){fetch('/graphql',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:document.getElementById('query').value})}).then(r=>r.json()).then(d=>document.getElementById('result').innerHTML='<pre>'+JSON.stringify(d,null,2)+'</pre>')}
function saveSession(){fetch('/api/session',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({data:document.getElementById('data').value})}).then(r=>r.json()).then(d=>document.getElementById('result').innerHTML='<p>'+JSON.stringify(d)+'</p>')}
function transfer(){fetch('/api/transfer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({amount:document.getElementById('amount').value})}).then(r=>r.json()).then(d=>document.getElementById('result').innerHTML='<p>'+JSON.stringify(d)+'</p>')}
function checkURL(){fetch('/api/check-url',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:document.getElementById('url').value})}).then(r=>r.json()).then(d=>document.getElementById('result').innerHTML='<p>'+JSON.stringify(d)+'</p>')}
function startOAuth(){window.location.href='/api/oauth-start'}
function postComment(){fetch('/api/comment',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({comment:document.getElementById('comment').value})}).then(r=>r.text()).then(d=>document.getElementById('result').innerHTML=d)}
function searchSQL(){fetch('/api/sql-search?q='+encodeURIComponent(document.getElementById('query').value)).then(r=>r.json()).then(d=>document.getElementById('result').innerHTML='<pre>'+JSON.stringify(d,null,2)+'</pre>')}
function showHint(cid,hid){fetch('/hint/'+cid+'/'+hid).then(()=>location.reload())}
</script>
</body></html>
'''

# Routes
@app.route('/')
def index():
    conn = sqlite3.connect('ctf.db')
    c = conn.cursor()
    solved = {i: False for i in range(1, 11)}
    for row in c.execute('SELECT challenge_id FROM progress WHERE solved=1').fetchall():
        solved[row[0]] = True
    conn.close()
    return render_template_string(MAIN_TEMPLATE, challenges=CHALLENGES, solved=solved)

@app.route('/challenge/<int:cid>', methods=['GET', 'POST'])
def challenge(cid):
    if cid not in CHALLENGES:
        return redirect('/')
    
    ch = CHALLENGES[cid]
    conn = sqlite3.connect('ctf.db')
    c = conn.cursor()
    c.execute('SELECT solved, hints_used FROM progress WHERE challenge_id=?', (cid,))
    row = c.fetchone()
    solved = row[0] if row else 0
    hints_used = row[1] if row else 0
    
    message = success = None
    if request.method == 'POST':
        flag = request.form.get('flag', '')
        if flag == ch['flag']:
            c.execute('INSERT OR REPLACE INTO progress VALUES (?, ?, 1, ?)', (cid, cid, hints_used))
            conn.commit()
            solved = 1
            message = f'Correct! Flag: {ch["flag"]}'
            success = True
        else:
            message = 'Incorrect flag!'
            success = False
    
    conn.close()
    return render_template_string(CHALLENGE_TEMPLATE, ch=ch, id=cid, solved=solved, 
                                 hints_used=hints_used, message=message, success=success, enumerate=enumerate)

@app.route('/hint/<int:cid>/<int:hid>')
def show_hint(cid, hid):
    conn = sqlite3.connect('ctf.db')
    c = conn.cursor()
    c.execute('SELECT hints_used FROM progress WHERE challenge_id=?', (cid,))
    row = c.fetchone()
    hints_used = row[0] if row else 0
    if hid >= hints_used:
        c.execute('INSERT OR REPLACE INTO progress VALUES (?, ?, 0, ?)', (cid, cid, hid+1))
        conn.commit()
    conn.close()
    return jsonify({'success': True})

# Challenge APIs
@app.route('/api/verify-token', methods=['POST'])
def verify_token():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    try:
        # Vulnerable: accepts alg=none
        decoded = jwt.decode(token, options={"verify_signature": False})
        if decoded.get('role') == 'admin':
            return jsonify({'success': True, 'flag': CHALLENGES[1]['flag']})
        return jsonify({'success': False, 'role': decoded.get('role')})
    except:
        return jsonify({'error': 'Invalid token'})

@app.route('/api/search')
def search():
    q = request.args.get('q', '')
    # Vulnerable SSTI
    if '{{' in q or '{%' in q:
        try:
            result = render_template_string(f"<h3>Search: {q}</h3><p>Found: User_{q}</p>")
            if 'CTF{' in result:
                return result
            return result
        except:
            return f"<p>Error in template: {q}</p>"
    return f"<p>Searching for: {q}</p>"

@app.route('/api/merge', methods=['POST'])
def merge_json():
    data = request.json
    # Vulnerable to prototype pollution (simulated)
    base = {'user': 'guest', 'role': 'user'}
    def merge(target, source):
        for key in source:
            if key == '__proto__' and 'isAdmin' in source[key]:
                return {'flag': CHALLENGES[3]['flag']}
            target[key] = source[key]
        return target
    result = merge(base, data)
    return jsonify(result)

@app.route('/graphql', methods=['POST'])
def graphql():
    query = request.json.get('query', '')
    # Vulnerable: introspection enabled
    if '__schema' in query or '__type' in query:
        return jsonify({
            'data': {
                'flag': CHALLENGES[4]['flag'],
                '__schema': {'types': [{'name': 'Query'}, {'name': 'SecretFlag'}]}
            }
        })
    return jsonify({'data': {'users': [{'name': 'user1'}, {'name': 'user2'}]}})

@app.route('/api/session', methods=['POST'])
def save_session():
    data = request.json.get('data', '')
    # Vulnerable pickle deserialization (simulated)
    if 'pickle' in data or '__reduce__' in data or 'os.system' in data:
        return jsonify({'flag': CHALLENGES[5]['flag'], 'message': 'Code executed!'})
    return jsonify({'session': base64.b64encode(data.encode()).decode()})

@app.route('/api/transfer', methods=['POST'])
def transfer():
    amount = int(request.json.get('amount', 0))
    # Vulnerable race condition (simulated)
    session['transfers'] = session.get('transfers', 0) + 1
    if session['transfers'] > 5:  # Multiple rapid requests
        return jsonify({'flag': CHALLENGES[6]['flag'], 'message': 'Race condition exploited!'})
    return jsonify({'balance': 1000 - amount, 'status': 'transferred'})

@app.route('/api/check-url', methods=['POST'])
def check_url():
    url = request.json.get('url', '')
    # Vulnerable SSRF
    if '169.254.169.254' in url or 'metadata' in url:
        return jsonify({'flag': CHALLENGES[7]['flag'], 'metadata': 'leaked'})
    return jsonify({'status': 'URL is alive'})

@app.route('/api/oauth-start')
def oauth_start():
    # Vulnerable OAuth
    token = 'oauth_token_' + os.urandom(8).hex()
    return redirect(f'/api/oauth-callback?token={token}&state=user123')

@app.route('/api/oauth-callback')
def oauth_callback():
    token = request.args.get('token')
    state = request.args.get('state')
    # Vulnerable: no CSRF check, token reuse
    if state != 'user123':
        return jsonify({'flag': CHALLENGES[8]['flag'], 'message': 'Token confusion!'})
    return jsonify({'access_token': token})

@app.route('/api/comment', methods=['POST'])
def comment():
    comment = request.json.get('comment', '')
    # Vulnerable DOM clobbering
    if '<a' in comment and 'id=' in comment and ('href=' in comment or 'name=' in comment):
        return f'<div>{comment}</div><p>Flag: {CHALLENGES[9]["flag"]}</p>'
    return f'<div>{comment}</div>'

@app.route('/api/sql-search')
def sql_search():
    q = request.args.get('q', '')
    # Vulnerable SQL injection with WAF bypass
    waf_blocked = ['union', 'select', 'or', 'and', '--', '#']
    if any(word in q.lower() for word in waf_blocked):
        # Check for bypass techniques
        if '/*!' in q or '0x' in q or q.count('/*') > 0:
            return jsonify({'flag': CHALLENGES[10]['flag'], 'message': 'WAF bypassed!'})
        return jsonify({'error': 'Blocked by WAF'})
    return jsonify({'users': ['user1', 'user2']})

if __name__ == '__main__':
    print(" CTF Platform starting...")
    print(" Visit: http://localhost:5000")
    print(" 10 Advanced Web Security Challenges")
    app.run(debug=True, host='0.0.0.0', port=5000)
