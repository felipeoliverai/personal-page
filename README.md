# Felipe Oliveira — Personal Site

Source code for my personal website ([felipe.dev](https://felipe.dev) — TBD).

I'm a Senior Machine Learning Engineer. This site hosts my bio, publications, talks, projects, and experience.

## Stack

- **[Hugo](https://gohugo.io/)** (extended) — static site generator
- **[Tailwind CSS v4](https://tailwindcss.com/)** — styling
- **[Preact](https://preactjs.com/)** — interactive components
- **[Pagefind](https://pagefind.app/)** — client-side search
- Built on top of the [HugoBlox Academic CV](https://github.com/HugoBlox/hugo-theme-academic-cv) template (MIT)

## Local development

Requires Hugo extended, Node.js, pnpm, and Go.

```bash
# Install dependencies
brew install hugo pnpm go
pnpm install

# Run the dev server (http://localhost:1313)
pnpm dev
```

## Build

```bash
pnpm build
```

The static site is output to `public/`.

## Content

All content lives under `content/` as Markdown:

- `content/authors/` — author profiles (data in `data/authors/me.yaml`)
- `content/publications/` — papers
- `content/events/` — talks
- `content/projects/` — projects
- `content/experience.md` — experience page

Site-wide config is in `config/_default/`.

## License

The site content (text, images) is © Felipe Oliveira.
The underlying template is MIT-licensed — see [LICENSE.md](LICENSE.md).
