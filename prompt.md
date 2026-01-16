# problem

## while dragging and dropping files into the dropzone, i get this error:
```
## Error Type
Console TypeError

## Error Message
Failed to fetch


    at proxyApplyFn.ApplyContext.reflect (chrome-extension://ddkjiahejlhfcafbddmgiahcphecmpfh/rulesets/scripting/scriptlet/main/ubol-tests.js:814:35)
    at <unknown> (chrome-extension://ddkjiahejlhfcafbddmgiahcphecmpfh/rulesets/scripting/scriptlet/main/ubol-tests.js:566:38)
    at Object.apply (chrome-extension://ddkjiahejlhfcafbddmgiahcphecmpfh/rulesets/scripting/scriptlet/main/ubol-tests.js:834:20)
    at <unknown> (file:///Users/logan/Developer/vibes/Effort-AIGI-Detection/frontend/.next/dev/static/chunks/_3c38d005._.js:1385:44)
    at Array.map (<anonymous>:null:null)
    at handleFilesSelected (file:///Users/logan/Developer/vibes/Effort-AIGI-Detection/frontend/.next/dev/static/chunks/_3c38d005._.js:1380:36)
    at Dropzone[handleDrop] (file:///Users/logan/Developer/vibes/Effort-AIGI-Detection/frontend/.next/dev/static/chunks/_3c38d005._.js:121:25)

Next.js version: 16.1.0 (Turbopack)```
```

## Looking at the browser console, i see a lot of these lines repeating:
```
Failed to load resource: net::ERR_INSUFFICIENT_RESOURCESUnderstand this error
installHook.js:1 TypeError: Failed to fetch
    at proxyApplyFn.ApplyContext.reflect (ubol-tests.js:814:35)
    at ubol-tests.js:566:38
    at Object.apply (ubol-tests.js:834:20)
    at page.tsx:108:34
    at Array.map (<anonymous>)
    at handleFilesSelected (page.tsx:99:30)
    at Dropzone[handleDrop] (Dropzone.tsx:87:17)
overrideMethod @ installHook.js:1Understand this error
predict:1  Failed to load resource: net::ERR_INSUFFICIENT_RESOURCESUnderstand this error
installHook.js:1 TypeError: Failed to fetch
    at proxyApplyFn.ApplyContext.reflect (ubol-tests.js:814:35)
    at ubol-tests.js:566:38
    at Object.apply (ubol-tests.js:834:20)
    at page.tsx:108:34
    at Array.map (<anonymous>)
    at handleFilesSelected (page.tsx:99:30)
    at Dropzone[handleDrop] (Dropzone.tsx:87:17)
overrideMethod @ installHook.js:1Understand this error
geist-latin.woff2:1  Failed to load resource: net::ERR_INSUFFICIENT_RESOURCESUnderstand this error
:3000/__nextjs_original-stack-frames:1  Failed to load resource: net::ERR_INSUFFICIENT_RESOURCES
```
## what's happening visually
It seems that the folder is getting uploaded fine, the images are still being processed.

## What's next
I want you to fix this error. 