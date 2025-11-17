# Release Setup Guide

## ✅ Completed

1. **Changeset Configuration**
   - Initialized changeset for version management
   - Created initial release changeset for v0.1.0
   - Configured for public package publishing

2. **GitHub Actions Workflows**
   - CI workflow: Runs tests, lint, type-check, build on push/PR
   - Release workflow: Automated publishing to npm when changeset is merged

3. **VitePress Documentation**
   - Configured VitePress with Axon branding
   - Created home page with features and quick example
   - Added Getting Started guide
   - Ready for Vercel deployment

4. **Vercel Configuration**
   - `vercel.json` configured for VitePress build
   - Build command: `bun run docs:build`
   - Output directory: `docs/.vitepress/dist`

## 📋 Manual Steps Required

### 1. GitHub Secrets Setup

Add these secrets in GitHub repository settings (Settings → Secrets and variables → Actions):

```
NPM_TOKEN=<your-npm-token>
```

**Get NPM Token:**
1. Login to npm: `npm login`
2. Generate token: `npm token create --type=automation`
3. Copy token and add as GitHub secret

### 2. Trigger Initial Release

The changeset workflow will create a "Version Packages" PR automatically when you push to main. To publish:

```bash
# The workflow will create a PR automatically
# Review the PR, then merge it
# On merge, packages will be published to npm automatically
```

**Or manually trigger version:**
```bash
bun run version  # Updates package.json versions
git add .
git commit -m "chore: release v0.1.0"
git push
# Release workflow will publish on push
```

### 3. Vercel Deployment Setup

**Option 1: Vercel CLI (Recommended)**
```bash
# Install Vercel CLI
bun add -g vercel

# Login
vercel login

# Deploy
vercel --prod
```

**Option 2: Vercel Dashboard**
1. Go to https://vercel.com/new
2. Import GitHub repository: `SylphxAI/Axon`
3. Configure:
   - Framework Preset: VitePress
   - Build Command: `bun run docs:build`
   - Output Directory: `docs/.vitepress/dist`
4. Deploy

**Set up automatic deployments:**
- Vercel will automatically deploy on push to main
- Preview deployments for PRs

### 4. npm Organization Setup

Ensure `@sylphx` scope is registered:

```bash
# Check if scope exists
npm org ls @sylphx

# If not exists, create it
# (Requires npm account)
```

### 5. Package Publishing Verification

After first publish, verify:

```bash
npm view @sylphx/tensor
npm view @sylphx/nn
# ... check all packages
```

## 📦 Package Structure

All packages configured for publishing:
- `@sylphx/tensor`
- `@sylphx/nn`
- `@sylphx/functional`
- `@sylphx/optim`
- `@sylphx/data`
- `@sylphx/wasm`
- `@sylphx/webgpu`
- `@sylphx/core`
- `@sylphx/predictors`

All at version `0.1.0` with public access.

## 🔄 Release Workflow

1. **Make changes** in feature branch
2. **Create changeset**: `bunx changeset`
3. **Commit and push**: PR will be created
4. **Merge PR**: CI tests run
5. **Changeset bot creates "Version Packages" PR**
6. **Review and merge**: Packages auto-publish to npm
7. **GitHub Release** created automatically

## 📚 Documentation Workflow

1. **Local development**: `bun run docs:dev`
2. **Preview build**: `bun run docs:build && bun run docs:preview`
3. **Push to main**: Vercel auto-deploys
4. **Custom domain** (optional): Configure in Vercel dashboard

## ⚠️ Important Notes

- **Never commit `.npmrc` with actual token** - Use environment variables
- **Test publishing** in a separate npm account first if unsure
- **Verify package contents** before first publish: `npm pack` in each package
- **Check bundle sizes**: All packages should be small (<100KB)

## 🎯 Next Steps

1. Add `NPM_TOKEN` to GitHub secrets
2. Wait for or manually trigger "Version Packages" PR
3. Merge PR to publish to npm
4. Deploy docs to Vercel
5. Announce release 🎉
