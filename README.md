## About

This project is used as the tensorflow backend to generate the `ml model for classification`.
Plan is to use this model with the ms/tfjs `messages_ms_forked`, which is the standard project for creating microservices.

> **note:** please contact [mimik](http://mimik.com/) for more info

## Build/Run

Dependencies are listed in `package.json`, simply run `npm install` to install the dependencies. Please note that
the `node_modules` folder, and the `package-lock.json` file, are both added to the `.gitignore` config.

## Commands

Following are useful commands for running this repo as a backend for generating ml model.

- `./package.sh` will create the `tar` for the project `messages_ms_backend`
- > scp messages_ms_backend.tar user1@192.168.1.24:/home/user1/tfjs_sample/
- > ssh user1@192.168.1.24 # pass1234
- once inside the system: `tar -xvf messages_ms_backend.tar`

## Using Model Locally

To use the model locally it should be copied from remote server, and then loaded using tfjs client
with the function `tf.loadLayersModel` (part of `"@tensorflow/tfjs": "2.3.0"` package).

Please note, to copy the model from remote server to local machine for testing with tfjs client use the following command.

>  scp user1@192.168.1.24:~/tfjs_sample/messages_ms_backend/model.tar ~/scp_data/

here, `~/scp_data/` is located on local machine

## Issues

Following are some critical issues with tensorflowjs setup (`tfjs-node`/`tfjs-node-gpu`).


1. Issue with tfjs-node-gpu
```js
// node-gpu
internal/process/esm_loader.js:74
    internalBinding('errors').triggerUncaughtException(
                              ^

Error: The Node.js native addon module (tfjs_binding.node) can not be found at path: node_modules/@tensorflow/tfjs-node-gpu/lib/napi-v5/tfjs_binding.node.    
Please run command 'npm rebuild @tensorflow/tfjs-node build-addon-from-source' to rebuild the native addon module.
If you have problem with building the addon module, please check https://github.com/tensorflow/tfjs/blob/master/tfjs-node/WINDOWS_TROUBLESHOOTING.md or file an issue.
    at Object.<anonymous> (node_modules/@tensorflow/tfjs-node-gpu/dist/index.js:49:11)
    at Module._compile (internal/modules/cjs/loader.js:999:30)
    at Object.Module._extensions..js (internal/modules/cjs/loader.js:1027:10)
    at Module.load (internal/modules/cjs/loader.js:863:32)
    at Function.Module._load (internal/modules/cjs/loader.js:708:14)
    at ModuleWrap.<anonymous> (internal/modules/esm/translators.js:188:29)
    at ModuleJob.run (internal/modules/esm/module_job.js:145:37)
    at async Loader.import (internal/modules/esm/loader.js:182:24)
    at async Object.loadESM (internal/process/esm_loader.js:68:5)
```

2. Issue with tfjs-node
```js
// node-cpu
internal/process/esm_loader.js:74
    internalBinding('errors').triggerUncaughtException(
                              ^

Error: The Node.js native addon module (tfjs_binding.node) can not be found at path: node_modules/@tensorflow/tfjs-node/lib/napi-v5/tfjs_binding.node.        
Please run command 'npm rebuild @tensorflow/tfjs-node build-addon-from-source' to rebuild the native addon module.
If you have problem with building the addon module, please check https://github.com/tensorflow/tfjs/blob/master/tfjs-node/WINDOWS_TROUBLESHOOTING.md or file an issue.
    at Object.<anonymous> (node_modules/@tensorflow/tfjs-node/dist/index.js:49:11)
    at Module._compile (internal/modules/cjs/loader.js:999:30)
    at Object.Module._extensions..js (internal/modules/cjs/loader.js:1027:10)
    at Module.load (internal/modules/cjs/loader.js:863:32)
    at Function.Module._load (internal/modules/cjs/loader.js:708:14)
    at ModuleWrap.<anonymous> (internal/modules/esm/translators.js:188:29)
    at ModuleJob.run (internal/modules/esm/module_job.js:145:37)
    at async Loader.import (internal/modules/esm/loader.js:182:24)
    at async Object.loadESM (internal/process/esm_loader.js:68:5)
```

This issue happens because of N-API. When working with tfjs (in nodejs), N-API is used to create bindings that allow TensorFlow's native C++ code to be executed within a nodejs environment. This is crucial for performance, as it enables tfjs to leverage the full power of TensorFlow's optimized computations.

> Please note that running `npm rebuild @tensorflow/tfjs-node* build-addon-from-source` does not fix this issue.

Steps am following right now:

>  sudo npm install -g node-gyp@9.1.0

`node-gyp` is a tool used to compile native add-ons for nodejs. It allows developers to write custom native code in C, C++, or other languages, and then compile that code into a binary module that can be loaded/used within nodejs applications.

https://github.com/tensorflow/tfjs/issues/8064#issuecomment-1845588111

> nvm install 18.16.1 && nvm use 18.16.1


```js
const tf = require("@tensorflow/tfjs-node-gpu");
const model = tf.sequential();
console.log(model);

// prints the following in console
/*
Successfully opened dynamic library libcuda.so.1
Successfully opened dynamic library libcurand.so.10
Successfully opened dynamic library libcudnn.so.8

<ref *1> Sequential ...
*/
```

#### Creating `start.json`

```bash
cat >> start.json << EOF
{
  "name": "$MS_IMAGE_NAME",
  "image": "$MS_IMAGE_NAME",
  "env": {
    "MCM.BASE_API_PATH": "/helloworld/v1/",
    "MCM.WEBSOCKET_SUPPORT": "true"
  }
}
EOF
```
