"""
Sample codebase for embedding strategy comparison.
30 functions across 5 modules: auth, database, cache, payment, notification.
Used as the knowledge base to search against.
"""

# ── Module: Authentication ─────────────────────────────────────────────────

def validate_jwt_token(token: str) -> dict:
    """Decode and validate a JWT token. Returns payload if valid."""
    import jwt
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return {"valid": True, "user_id": payload["sub"], "role": payload.get("role")}
    except jwt.ExpiredSignatureError:
        return {"valid": False, "error": "Token expired"}
    except jwt.InvalidTokenError:
        return {"valid": False, "error": "Invalid token"}


def hash_password(plain_password: str) -> str:
    """Hash a plaintext password using bcrypt. Use this before storing."""
    import bcrypt
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(plain_password.encode(), salt).decode()


def verify_password(plain_password: str, hashed: str) -> bool:
    """Check if a plaintext password matches its bcrypt hash."""
    import bcrypt
    return bcrypt.checkpw(plain_password.encode(), hashed.encode())


def generate_access_token(user_id: str, role: str, expires_in: int = 3600) -> str:
    """Generate a signed JWT access token with user_id, role, and expiry."""
    import jwt, time
    payload = {"sub": user_id, "role": role, "iat": time.time(), "exp": time.time() + expires_in}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def refresh_token_rotation(old_refresh_token: str) -> dict:
    """Invalidate old refresh token and issue a new token pair (access + refresh)."""
    token_data = decode_refresh_token(old_refresh_token)
    invalidate_token(old_refresh_token)
    new_access = generate_access_token(token_data["user_id"], token_data["role"])
    new_refresh = generate_refresh_token(token_data["user_id"])
    return {"access_token": new_access, "refresh_token": new_refresh}


def check_permission(user_role: str, required_permission: str) -> bool:
    """Verify that a user role has the specified permission. Uses RBAC table."""
    ROLE_PERMISSIONS = {
        "admin": ["read", "write", "delete", "manage_users"],
        "editor": ["read", "write"],
        "viewer": ["read"],
    }
    return required_permission in ROLE_PERMISSIONS.get(user_role, [])


# ── Module: Database ───────────────────────────────────────────────────────

def get_db_connection(pool_size: int = 10):
    """Create a PostgreSQL connection pool. Returns a connection pool object."""
    from sqlalchemy import create_engine
    engine = create_engine(DATABASE_URL, pool_size=pool_size, pool_pre_ping=True)
    return engine.connect()


def execute_query(query: str, params: dict = None) -> list:
    """Execute a parameterized SQL query safely. Returns list of row dicts."""
    with get_db_connection() as conn:
        result = conn.execute(query, params or {})
        return [dict(row) for row in result.fetchall()]


def bulk_insert(table: str, records: list[dict]) -> int:
    """Insert multiple rows in a single transaction. Returns inserted count."""
    if not records:
        return 0
    with get_db_connection() as conn:
        with conn.begin():
            conn.execute(f"INSERT INTO {table} VALUES :data", {"data": records})
    return len(records)


def run_migration(migration_file: str) -> None:
    """Apply a SQL migration file. Records execution in migration_history table."""
    with open(migration_file) as f:
        sql = f.read()
    with get_db_connection() as conn:
        conn.execute(sql)
        conn.execute("INSERT INTO migration_history (filename) VALUES (?)", (migration_file,))


def paginate_query(query: str, page: int, page_size: int = 20) -> dict:
    """Execute a query with LIMIT/OFFSET pagination. Returns data and total count."""
    offset = (page - 1) * page_size
    total = execute_query(f"SELECT COUNT(*) FROM ({query}) t")[0]["count"]
    data = execute_query(f"{query} LIMIT {page_size} OFFSET {offset}")
    return {"data": data, "total": total, "page": page, "page_size": page_size}


def soft_delete_record(table: str, record_id: int) -> bool:
    """Mark a record as deleted without removing it. Sets deleted_at timestamp."""
    rows = execute_query(
        f"UPDATE {table} SET deleted_at = NOW() WHERE id = ? AND deleted_at IS NULL",
        {"id": record_id}
    )
    return len(rows) > 0


# ── Module: Cache ──────────────────────────────────────────────────────────

def get_cached_value(key: str):
    """Retrieve a value from Redis cache. Returns None if key doesn't exist."""
    import redis
    r = redis.Redis.from_url(REDIS_URL)
    value = r.get(key)
    return value.decode() if value else None


def set_cache(key: str, value: str, ttl_seconds: int = 3600) -> bool:
    """Store a value in Redis with an expiry time. Returns True on success."""
    import redis
    r = redis.Redis.from_url(REDIS_URL)
    return r.setex(key, ttl_seconds, value)


def invalidate_cache_pattern(pattern: str) -> int:
    """Delete all Redis keys matching a glob pattern. Returns count deleted."""
    import redis
    r = redis.Redis.from_url(REDIS_URL)
    keys = r.keys(pattern)
    if keys:
        r.delete(*keys)
    return len(keys)


def cache_user_session(user_id: str, session_data: dict, ttl: int = 86400) -> str:
    """Store user session data in Redis. Returns the session key."""
    import json
    key = f"session:{user_id}"
    set_cache(key, json.dumps(session_data), ttl)
    return key


def rate_limit_check(user_id: str, action: str, max_calls: int, window_seconds: int) -> bool:
    """Check if user has exceeded rate limit for an action. Returns True if allowed."""
    import redis
    r = redis.Redis.from_url(REDIS_URL)
    key = f"ratelimit:{user_id}:{action}"
    pipe = r.pipeline()
    pipe.incr(key)
    pipe.expire(key, window_seconds)
    count, _ = pipe.execute()
    return count <= max_calls


# ── Module: Payment ────────────────────────────────────────────────────────

def create_payment_intent(amount_cents: int, currency: str, user_id: str) -> dict:
    """Create a Stripe payment intent. Returns client_secret for frontend use."""
    import stripe
    intent = stripe.PaymentIntent.create(
        amount=amount_cents, currency=currency,
        metadata={"user_id": user_id}
    )
    return {"payment_intent_id": intent.id, "client_secret": intent.client_secret}


def process_refund(payment_intent_id: str, amount_cents: int = None) -> dict:
    """Issue a refund for a payment. Pass amount for partial refund, None for full."""
    import stripe
    params = {"payment_intent": payment_intent_id}
    if amount_cents:
        params["amount"] = amount_cents
    refund = stripe.Refund.create(**params)
    return {"refund_id": refund.id, "status": refund.status, "amount": refund.amount}


def calculate_order_total(items: list[dict], discount_code: str = None) -> dict:
    """Sum item prices, apply discount, compute tax. Returns breakdown dict."""
    subtotal = sum(item["price"] * item["quantity"] for item in items)
    discount = apply_discount_code(discount_code, subtotal) if discount_code else 0
    tax = (subtotal - discount) * TAX_RATE
    return {"subtotal": subtotal, "discount": discount, "tax": tax,
            "total": subtotal - discount + tax}


def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify Stripe webhook signature to prevent spoofed callbacks."""
    import stripe
    try:
        stripe.Webhook.construct_event(payload, signature, secret)
        return True
    except stripe.error.SignatureVerificationError:
        return False


def get_payment_history(user_id: str, limit: int = 20) -> list:
    """Retrieve paginated payment history for a user from the database."""
    return execute_query(
        "SELECT * FROM payments WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        {"user_id": user_id, "limit": limit}
    )


# ── Module: Notification ───────────────────────────────────────────────────

def send_email(to: str, subject: str, html_body: str, text_body: str = None) -> bool:
    """Send an email via SMTP. Returns True on success."""
    import smtplib
    from email.mime.multipart import MIMEMultipart
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["To"] = to
    # add parts...
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.sendmail(SMTP_FROM, to, msg.as_string())
    return True


def send_push_notification(device_token: str, title: str, body: str, data: dict = None) -> dict:
    """Send a Firebase push notification to a mobile device."""
    import firebase_admin.messaging as messaging
    message = messaging.Message(
        notification=messaging.Notification(title=title, body=body),
        data=data or {},
        token=device_token,
    )
    response = messaging.send(message)
    return {"message_id": response, "success": True}


def send_sms(phone_number: str, message: str) -> dict:
    """Send an SMS via Twilio. Returns delivery status."""
    from twilio.rest import Client
    client = Client(TWILIO_SID, TWILIO_TOKEN)
    msg = client.messages.create(body=message, from_=TWILIO_FROM, to=phone_number)
    return {"sid": msg.sid, "status": msg.status}


def batch_notify_users(user_ids: list[str], notification: dict) -> dict:
    """Send the same notification to multiple users. Returns success/failure counts."""
    results = {"sent": 0, "failed": 0, "errors": []}
    for uid in user_ids:
        try:
            user = get_user_by_id(uid)
            send_email(user["email"], notification["subject"], notification["body"])
            results["sent"] += 1
        except Exception as e:
            results["failed"] += 1
            results["errors"].append({"user_id": uid, "error": str(e)})
    return results


def schedule_notification(user_id: str, notification: dict, send_at: str) -> str:
    """Queue a notification for future delivery. Returns task ID."""
    task = {
        "user_id": user_id,
        "notification": notification,
        "send_at": send_at,
        "status": "pending"
    }
    task_id = create_task(task)
    return task_id
