# 🏥 DermaSafe — AI Dermatology Triage Platform

<div align="center">

[![Live Demo](https://img.shields.io/badge/Live-dermasafe.app-blue?style=flat-square&logo=globe)](https://dermasafe.app)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production-brightgreen?style=flat-square)]()

**AI-powered dermatology triage with clinician oversight, secure authentication, and self-hosted infrastructure**

</div>

---

## 🎯 Overview

DermaSafe is a **production-grade healthcare AI platform** that combines:
- 🤖 **AI-powered risk prediction** via Derm Foundation model for skin lesion analysis
- 🧠 **Conversational AI** with Gemini for result interpretation
- 👨‍⚕️ **Clinician review workflow** with PII-controlled access
- 🔐 **Security-first architecture** with JWT + TOTP 2FA, RBAC, and audit logging
- 🏠 **Self-hosted deployment** for complete data ownership

**Not just a model** — DermaSafe is an **end-to-end engineered system** where security, privacy, and clinician oversight are built into the architecture from day one.

---

## ✨ Key Features

### 👥 User Features
- ✅ Secure registration with email verification
- ✅ Skin lesion image upload with preprocessing
- ✅ AI risk prediction via Derm Foundation model
- ✅ Gemini-powered AI assistant for result Q&A
- ✅ Prediction history and medical records access

### 👨‍⚕️ Clinician Features
- ✅ Secure clinician queue for case review
- ✅ Controlled PII access with audit logging
- ✅ Case notes and recommendation workflow
- ✅ Patient communication interface

### 🛡️ Security & Compliance
- ✅ **JWT Authentication** with refresh token rotation
- ✅ **TOTP 2FA** (Time-based One-Time Password)
- ✅ **Role-Based Access Control (RBAC)** — User, Clinician, Admin
- ✅ **Audit Logging** — All PII access tracked and logged
- ✅ **bcrypt Password Hashing** — Secure credential storage
- ✅ **CSRF Protection** — Token-based anti-CSRF
- ✅ **CSP & HSTS** — Content Security Policy, Strict Transport Security
- ✅ **Rate Limiting** — DDoS/brute-force protection
- ✅ **Reverse Proxy Hardening** — Secure API gateway

### 🎛️ Admin Dashboard
- ✅ User and role management
- ✅ Credit/tier system for feature access
- ✅ Clinician assignment and approval
- ✅ Prediction statistics and insights
- ✅ System configuration and secrets management

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                          │
│         React + TanStack Start + Tailwind CSS               │
│                   (Port 8080)                                │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTPS (JWT Auth)
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    API Gateway / Backend                     │
│         Node.js / Express.js (Port 5000)                    │
│    ├─ Authentication & RBAC                                 │
│    ├─ Prediction Management                                 │
│    ├─ Clinician Queue                                       │
│    ├─ Audit Logging                                         │
│    └─ Admin Operations                                      │
└──┬─────────────────┬─────────────────────┬──────────────────┘
   │                 │                     │
   │                 │                     │
   ▼                 ▼                     ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────────┐
│ PostgreSQL  │  │  ML Sidecar  │  │ Gemini AI        │
│ Database    │  │  (FastAPI)   │  │ Assistant        │
│             │  │  Port 8000   │  │                  │
│ • Users     │  │              │  │ • Interpretation │
│ • Roles     │  │ • Derm Model │  │ • Q&A            │
│ • Sessions  │  │ • Inference  │  │ • Guidance       │
│ • Cases     │  │ • Validation │  │                  │
│ • Audit Log │  │              │  │                  │
└─────────────┘  └──────────────┘  └──────────────────┘
```

### Service Breakdown

| Service | Technology | Purpose |
|---|---|---|
| **Frontend** | React 18, TypeScript, TanStack Start, Tailwind | User interface, authenticated requests |
| **Backend API** | Node.js, Express.js | Business logic, auth, RBAC, audit |
| **ML Inference** | Python, FastAPI | Derm model inference, image preprocessing |
| **Database** | PostgreSQL | Persistent data, audit trails |
| **AI Assistant** | Gemini API | Conversational result interpretation |
| **Deployment** | Docker Compose | Orchestration and self-hosted deployment |

---

## 📊 Database Schema

```sql
-- Core Tables
users (id, email, password_hash, full_name, created_at, verified_at)
roles (id, name, permissions_json) -- User, Clinician, Admin
user_roles (user_id, role_id)
refresh_tokens (user_id, token_hash, expires_at)

-- Predictions & Cases
predictions (id, user_id, image_path, prediction_score, model_version, created_at)
clinician_cases (id, prediction_id, clinician_id, status, notes, created_at)

-- Audit & Security
audit_logs (id, user_id, action, resource, timestamp, ip_address)
access_logs (id, user_id, accessed_resource, pii_accessed, timestamp)
```

---

## 🔐 Security Highlights

### Authentication Flow
```
User Input → Registration (email + password)
           → Email Verification
           → Login (email + password)
           → TOTP 2FA Setup & Verification
           → JWT + Refresh Token Issued
           → Authenticated API Requests (JWT in Authorization header)
           → Refresh Token Rotation (auto-renewed on expiry)
```

### Authorization (RBAC)
```
Roles:
  • User: Can upload, view own predictions, interact with AI
  • Clinician: Can review assigned cases, add notes, access PII with audit
  • Admin: Full system access, user management, configuration

Permissions:
  • User → ["upload_image", "view_own_predictions", "chat_assistant"]
  • Clinician → ["review_cases", "access_pii", "add_notes"]
  • Admin → ["manage_users", "manage_roles", "view_audit_logs", "system_config"]
```

### PII Access Control
- **All PII access is logged** (user ID, timestamp, resource, IP address)
- **Clinicians cannot download/export patient data** — view-only with audit trail
- **Admins can audit all access** via admin dashboard
- **Sensitive operations** (role promotion, credential changes) require additional verification

---

## 🚀 Deployment & Self-Hosting

### Prerequisites
- Docker & Docker Compose
- 4GB+ RAM, 20GB+ disk space
- Environment configuration (SMTP, secrets, etc.)

### Quick Start (Self-Hosted)

```bash
# 1. Clone repository
git clone https://github.com/ravi5775/dermasafe.git
cd dermasafe

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings:
# - DATABASE_URL
# - JWT_SECRET, REFRESH_SECRET
# - TOTP_ISSUER
# - SMTP_* (email settings)
# - GEMINI_API_KEY
# - DEFAULT_ADMIN_EMAIL, DEFAULT_ADMIN_PASSWORD

# 3. Start all services
docker-compose up -d

# 4. Access application
# Frontend: http://localhost:8080
# Admin: http://localhost:8080/admin
```

### Services Running
```yaml
frontend:
  image: dermasafe-frontend:latest
  ports: ["8080:3000"]

backend:
  image: dermasafe-backend:latest
  ports: ["5000:5000"]
  depends_on: [postgres, ml-sidecar]

ml-sidecar:
  image: dermasafe-ml:latest
  ports: ["8000:8000"]

postgres:
  image: postgres:15
  volumes: [postgres_data:/var/lib/postgresql/data]
```

### Production Deployment
- Use **Cloudflare** or **nginx** as reverse proxy with SSL
- Configure **firewall rules** (allow only frontend/backend ports)
- Enable **automatic backups** of PostgreSQL database
- Monitor **logs** for security incidents
- Rotate **secrets** regularly (JWT keys, API tokens)

---

## 📈 Performance Metrics

| Metric | Target | Status |
|---|---|---|
| **API Response Time** | < 200ms | ✅ Achieved |
| **Image Processing** | < 3s | ✅ Achieved |
| **Model Inference** | < 5s | ✅ Achieved |
| **Concurrent Users** | 100+ | ✅ Tested |
| **Database Query** | < 50ms | ✅ Optimized |

---

## 🛠️ Tech Stack

### Frontend
```
React 18 + TypeScript
├─ TanStack Start (Full-stack React)
├─ Tailwind CSS (Styling)
├─ React Query (Data fetching)
├─ Zustand (State management)
└─ Zod (Form validation)
```

### Backend
```
Node.js + Express
├─ JWT for authentication
├─ TOTP for 2FA
├─ bcrypt for password hashing
├─ Morgan for logging
├─ CORS & CSRF middleware
└─ Rate limiting (express-rate-limit)
```

### ML Service
```
Python + FastAPI
├─ Derm Foundation Model
├─ OpenCV (image preprocessing)
├─ NumPy & Scikit-image
└─ Pydantic (validation)
```

### Database & Storage
```
PostgreSQL 15
├─ JSONB for flexible configs
├─ Indexes for audit queries
└─ Partitioning for large tables
```

### Infrastructure
```
Docker & Docker Compose
├─ Multi-stage builds
├─ Volume management
└─ Environment-based config
```

---

## 📚 API Endpoints

### Authentication
```
POST   /api/auth/register        → Create account
POST   /api/auth/login           → Login + TOTP setup
POST   /api/auth/verify-otp      → Verify 2FA code
POST   /api/auth/refresh         → Refresh JWT token
POST   /api/auth/logout          → Revoke refresh token
```

### Predictions
```
POST   /api/predictions          → Upload image + get prediction
GET    /api/predictions          → List user's predictions
GET    /api/predictions/:id      → Get prediction details
```

### AI Assistant
```
POST   /api/assistant/chat       → Send message to Gemini
GET    /api/assistant/context/:pred_id → Get conversation history
```

### Clinician (Protected)
```
GET    /api/clinician/queue      → List cases assigned to clinician
GET    /api/clinician/cases/:id  → Get case details (PII logged)
POST   /api/clinician/cases/:id/notes → Add clinical notes
```

### Admin (Protected)
```
GET    /api/admin/users          → List users
POST   /api/admin/users/:id/role → Assign role
GET    /api/admin/audit-logs     �� View audit trail
GET    /api/admin/system/stats   → System statistics
```

---

## 🧪 Testing

### Run Tests
```bash
# Backend tests
npm test --prefix backend

# Frontend tests
npm test --prefix frontend

# ML service tests
pytest ml-service/tests/
```

### Manual Testing Checklist
- [ ] User registration & email verification
- [ ] Login with TOTP 2FA
- [ ] Image upload & prediction
- [ ] AI assistant conversation
- [ ] Clinician case review
- [ ] Admin dashboard operations
- [ ] Audit log verification
- [ ] JWT token rotation

---

## 🔄 Development Workflow

```bash
# 1. Start development environment
docker-compose -f docker-compose.dev.yml up

# 2. Frontend development
cd frontend && npm run dev

# 3. Backend development
cd backend && npm run dev

# 4. ML service development
cd ml-service && python -m uvicorn main:app --reload

# 5. Test in browser
# Frontend: http://localhost:3000
# Backend: http://localhost:5000
# ML: http://localhost:8000/docs
```

---

## 🎓 Lessons Learned

### Architecture
- **Separating ML inference into its own service** made the system dramatically easier to reason about, secure, and scale independently of the main API
- **Security and audit design must be part of initial architecture**, not retrofitted
- **Multi-service architecture pays off** even for smaller projects when different parts have different requirements (web, inference, data)

### Healthcare AI
- **Model accuracy alone isn't enough** — clinical systems need oversight, audit trails, and human review
- **Data ownership and privacy** are non-negotiable for healthcare platforms
- **Clinician workflows** must be part of the product design, not an afterthought

### Security
- **JWT refresh token rotation** is crucial for long-lived sessions
- **TOTP 2FA** is the practical standard for healthcare apps
- **Audit logging** needs to be comprehensive but not noisy

---

## 🚀 Future Roadmap

### Phase 2
- [ ] Mobile app (React Native / Flutter)
- [ ] FHIR integration for EHR interoperability
- [ ] Batch prediction API for institutional use
- [ ] Explainability output (SHAP/LIME) alongside predictions

### Phase 3
- [ ] Multi-model ensemble for improved accuracy
- [ ] Expanded image analysis (full-body dermatology)
- [ ] Telemedicine integration (video consultation)
- [ ] Advanced analytics dashboard for dermatologists

### Phase 4
- [ ] International compliance (GDPR, HIPAA, PIPEDA)
- [ ] Insurance claim integration
- [ ] Clinical research dataset export
- [ ] Open-source model fine-tuning pipeline

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details

---

## 📧 Contact & Support

**Author:** Adabala Pavan  
**Email:** [hello@pavanadabala.me](mailto:hello@pavanadabala.me)  
**Portfolio:** [pavanadabala.me](https://pavanadabala.me)  
**GitHub:** [@ravi5775](https://github.com/ravi5775)

---

## 🙏 Acknowledgments

- **Derm Foundation Model** — Medical image classification
- **Google Gemini API** — Conversational AI
- **Open-source community** — FastAPI, Express, PostgreSQL, Docker

---

<div align="center">

**Building healthcare AI systems where security, privacy, and clinician oversight are first-class concerns.**

Made with ❤️ by [Adabala Pavan](https://github.com/ravi5775)

</div>
