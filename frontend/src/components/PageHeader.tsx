import type { ReactNode } from 'react'

export default function PageHeader({ eyebrow, title, text, action }: { eyebrow: string; title: string; text: string; action?: ReactNode }) {
  return <header className="page-header"><div><p className="eyebrow">{eyebrow}</p><h1>{title}</h1><p>{text}</p></div>{action}</header>
}

