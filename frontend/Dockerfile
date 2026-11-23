# ---------- 前端多阶段镜像：Node 构建 dist + Nginx 静态托管（M5 容器编排） ----------
# 阶段一：用 Node 构建 Vue 产物
FROM node:22-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

# 阶段二：Nginx 只托管构建后的静态文件，体积更小
FROM nginx:1.27-alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
