$(function () {
    var $copyIcon = $('<i class="fas fa-copy code_copy" title="复制代码" aria-hidden="true"></i>');
    var $notice = $('<div class="codecopy_notice"></div>');

    $('.code-area').prepend($copyIcon).prepend($notice);

    function getCodeText($pre) {
        var lines = $pre.find('.line').map(function () {
            return $(this).text();
        }).get();

        if (lines.length > 0) {
            return lines.join('\n');
        }

        var $code = $pre.find('code');
        if ($code.length > 0) {
            return $code.text();
        }

        return $pre.text();
    }

    function showNotice(ctx, text) {
        $(ctx).prev('.codecopy_notice')
            .text(text)
            .animate({
                opacity: 1,
                top: 30
            }, 450, function () {
                var notice = this;
                setTimeout(function () {
                    $(notice).animate({
                        opacity: 0,
                        top: 0
                    }, 650);
                }, 400);
            });
    }

    function fallbackCopy(text) {
        var textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.setAttribute('readonly', '');
        textarea.style.position = 'fixed';
        textarea.style.top = '-9999px';
        document.body.appendChild(textarea);
        textarea.select();

        var ok = document.execCommand('copy');
        document.body.removeChild(textarea);
        return ok;
    }

    $('.code-area .fa-copy').on('click', function () {
        var $pre = $(this).siblings('pre');
        var text = getCodeText($pre);
        var ctx = this;

        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(function () {
                showNotice(ctx, '复制成功');
            }).catch(function () {
                showNotice(ctx, fallbackCopy(text) ? '复制成功' : '复制失败');
            });
            return;
        }

        showNotice(ctx, fallbackCopy(text) ? '复制成功' : '复制失败');
    });
});
