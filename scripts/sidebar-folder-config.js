'use strict';
const fs   = require('fs');
const path = require('path');

/**
 * Scan _posts/ for config.json files and expose as site.sidebarFolderConfigs.
 * Key: relative path from _posts root, e.g. "ACM-ICPC/Dynamic_Programming"
 * Value: parsed config.json object, e.g. { name: "动态规划" }
 */
hexo.locals.set('sidebarFolderConfigs', function () {
    const postsDir = path.join(hexo.source_dir, '_posts');
    const configs  = {};

    function walk(dir) {
        if (!fs.existsSync(dir)) return;
        for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
            const full = path.join(dir, entry.name);
            if (entry.isDirectory()) {
                walk(full);
            } else if (entry.name === 'config.json') {
                try {
                    const cfg = JSON.parse(fs.readFileSync(full, 'utf8'));
                    const rel = path.relative(postsDir, dir).replace(/\\/g, '/');
                    configs[rel] = cfg;
                } catch (e) { /* ignore malformed config */ }
            }
        }
    }

    walk(postsDir);
    return configs;
});
