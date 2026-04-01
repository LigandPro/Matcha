import type { Metadata } from 'next';
import '../src/styles/app.scss';

export const metadata: Metadata = {
  title: 'Matcha UI',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <div id="root">{children}</div>
      </body>
    </html>
  );
}
