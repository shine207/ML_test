import re
  
pattern = 'gibo_load\(\'[^\']*\''
#pattern = 'gibo_load\('
#pattern = 'href'
text = """<tr>
                                        <td>1939.09.30</td>
                                        <td class="tit">???? vs ??? 10?? ?1?</td>
                                        <td><a href="javascript:gisa_popup2('???')">???</a></td>
                                        <td><a href="javascript:gisa_popup2('??? ???')">??? ???</a></td>
                                        <td>276? ?2??</td>
                                        <td>
                                                <a href="javascript:" onclick="javascript:gibo_load('http://open.cyberoro.com/gibo/test/19390930ten-g01.sgf', '468')"><img src="/images/btn/btn_html5.gif" alt="Html5" /></a>
                                                <a href="javascript:" onclick="gibo_load_activex('http://open.cyberoro.com/gibo/test/19390930ten-g01.sgf')"><img src="/images/btn/btn_activex.gif" alt="ActiveX" /></a>                                         
                                        </td>
                                </tr>
                                
                                <tr>
                                        <td>1938.06.26</td>
                                        <td class="tit">??? ??? ???</td>
                                        <td><a href="javascript:gisa_popup2('??? ???')">??? ???</a></td>
                                        <td><a href="javascript:gisa_popup2('???')">???</a></td>
                                        <td>237? ?5??</td>
                                        <td>
                                                <a href="javascript:" onclick="javascript:gibo_load('http://open.cyberoro.com/gibo/test/19380626niho-sg01.sgf', '315')"><img src="/images/btn/btn_html5.gif" alt="Html5" /></a>
                                                <a href="javascript:" onclick="gibo_load_activex('http://open.cyberoro.com/gibo/test/19380626niho-sg01.sgf')"><img src="/images/btn/btn_activex.gif" alt="ActiveX" /></a>                                               
                                        </td>
                                </tr>"""
 
r = re.compile(pattern)
 
matches = r.finditer(text)
for m in matches:
    startIndex = m.start()
    endIndex = m.end()
    print(text[startIndex+11:endIndex-1])
