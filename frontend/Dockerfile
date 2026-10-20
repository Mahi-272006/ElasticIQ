# Standalone dashboard image for development against an external API.
# The production path embeds this build inside the root Dockerfile instead.
FROM node:22-alpine

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . ./

EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
