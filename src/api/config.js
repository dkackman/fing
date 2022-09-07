const yaml = require('js-yaml');
const fs = require('fs');
const os = require('os');
const path = require('path');

let _config;

module.exports = {
    loadConfig() {
        const configRootDir = getConfigRootDir();

        _config = yaml.load(
            fs.readFileSync(path.resolve(configRootDir, 'config.yaml'), 'utf8'),
        );
    },

    get configYaml() { return _config; }
};

function getConfigRootDir() {
    const homedir = os.homedir();

    return 'GENERATOR_ROOT' in process.env
        ? untildify(process.env.GENERATOR_ROOT)
        : path.join(homedir, '.fing');
}

function untildify(pathWithTilde) {
    if (typeof pathWithTilde !== 'string') {
        throw new TypeError(`Expected a string, got ${typeof pathWithTilde}`);
    }

    return homeDirectory
        ? pathWithTilde.replace(/^~(?=$|\/|\\)/, homeDirectory)
        : pathWithTilde;
}
