import { createHash } from 'node:crypto';
import { chmod, mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { delimiter, dirname, join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const rootDir = join(dirname(fileURLToPath(import.meta.url)), '..');
const hexoCli = join(rootDir, 'node_modules', 'hexo', 'bin', 'hexo');
const pandocVersion = '3.9.0.2';
const pandocReleases = {
  x64: {
    archive: `pandoc-${pandocVersion}-linux-amd64.tar.gz`,
    sha256: 'a69abfababda8a56969a254b09f9553a7be89ddec00d4e0fe9fd585d71a67508'
  },
  arm64: {
    archive: `pandoc-${pandocVersion}-linux-arm64.tar.gz`,
    sha256: 'b6d21e8f9c3b15744f5a7ab40248019157ed7793875dbe0383d4c82ff572b528'
  }
};

function run(command, args, options = {}) {
  const result = spawnSync(command, args, options);

  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(`${command} exited with status ${result.status}`);
  }

  return result;
}

function isExecutable(command) {
  try {
    run(command, ['--version'], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

async function installPandoc() {
  if (process.platform !== 'linux' || !pandocReleases[process.arch]) {
    throw new Error(
      `Pandoc is not installed, and automatic installation is unsupported on ${process.platform}/${process.arch}.`
    );
  }

  const release = pandocReleases[process.arch];
  const installDir = join(rootDir, '.cache', `pandoc-${pandocVersion}`);
  const executable = join(installDir, 'bin', 'pandoc');

  if (isExecutable(executable)) {
    return executable;
  }

  const temporaryDir = await mkdtemp(join(tmpdir(), 'blog-pandoc-'));
  const archivePath = join(temporaryDir, release.archive);
  const downloadUrl = `https://github.com/jgm/pandoc/releases/download/${pandocVersion}/${release.archive}`;

  try {
    console.log(`Pandoc not found; downloading ${release.archive}...`);
    const response = await fetch(downloadUrl);
    if (!response.ok) {
      throw new Error(`Failed to download Pandoc: HTTP ${response.status}`);
    }

    const archive = Buffer.from(await response.arrayBuffer());
    const digest = createHash('sha256').update(archive).digest('hex');
    if (digest !== release.sha256) {
      throw new Error(`Pandoc checksum mismatch: expected ${release.sha256}, received ${digest}`);
    }

    await writeFile(archivePath, archive);
    await rm(installDir, { recursive: true, force: true });
    await mkdir(installDir, { recursive: true });
    run('tar', ['-xzf', archivePath, '--strip-components=1', '-C', installDir], {
      stdio: 'inherit'
    });
    await chmod(executable, 0o755);
  } finally {
    await rm(temporaryDir, { recursive: true, force: true });
  }

  return executable;
}

let pandocPath = 'pandoc';
if (!isExecutable(pandocPath)) {
  pandocPath = await installPandoc();
}

console.log(`Using Pandoc at ${pandocPath}`);
const env = {
  ...process.env,
  PATH: `${dirname(pandocPath)}${delimiter}${process.env.PATH ?? ''}`
};
const result = spawnSync(process.execPath, [hexoCli, 'generate'], {
  cwd: rootDir,
  env,
  stdio: 'inherit'
});

if (result.error) {
  throw result.error;
}
process.exitCode = result.status ?? 1;
